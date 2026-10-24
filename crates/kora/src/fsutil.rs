//! Crash-safe file replacement for persistent caches.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, ErrorCode, Result, Status};

/// Atomically replaces `path` with `contents`: writes a uniquely named sibling temporary file,
/// flushes it to disk, renames it over the destination, and (on POSIX) syncs the directory.
/// Readers observe either the old or the new file, never a partial one.
pub fn write_atomic(path: &Path, contents: &[u8]) -> Status {
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    let io = |what: &str, error: std::io::Error| Error::new(ErrorCode::IoError, format!("{what}: {error}"));
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).map_err(|e| io("cannot create cache directory", e))?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = std::path::PathBuf::from(temporary);
    let written = (|| -> std::io::Result<()> {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()
    })();
    if let Err(error) = written {
        let _ = fs::remove_file(&temporary);
        return Err(io("cannot write temporary cache file", error));
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(io("cannot atomically publish cache file", error));
    }
    #[cfg(unix)]
    if let Ok(directory) = fs::File::open(parent.unwrap_or(Path::new("."))) {
        let _ = directory.sync_all();
    }
    Ok(())
}

/// Reads a file of at most `limit` bytes.
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)
        .map_err(|e| Error::new(ErrorCode::IoError, format!("cannot stat {}: {e}", path.display())))?;
    if metadata.len() == 0 || metadata.len() > limit {
        return Err(Error::new(
            ErrorCode::IoError,
            format!("{} is empty or exceeds {limit} bytes", path.display()),
        ));
    }
    let bytes = fs::read(path)
        .map_err(|e| Error::new(ErrorCode::IoError, format!("cannot read {}: {e}", path.display())))?;
    if bytes.len() as u64 > limit {
        return Err(Error::new(
            ErrorCode::IoError,
            format!("{} grew beyond {limit} bytes", path.display()),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
pub(crate) fn temp_dir(name: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!("kora-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    directory
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_existing_files_and_creates_parents() {
        let directory = temp_dir("atomic");
        let path = directory.join("nested/file.json");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two");
        let leftovers = fs::read_dir(path.parent().unwrap()).unwrap().count();
        assert_eq!(leftovers, 1);
        assert_eq!(read_bounded(&path, 3).unwrap(), b"two");
        assert!(read_bounded(&path, 2).is_err());
        assert!(read_bounded(&directory.join("missing"), 10).is_err());
        let _ = fs::remove_dir_all(directory);
    }
}
