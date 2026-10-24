# kora-worker

Runs dispatches for a remote [`Cluster`](../guides/cluster.md) on the machine where it runs.

```sh
kora-worker [--listen HOST:PORT] [--device INDEX] [--cache-mb N] [--name NAME]
```

| Option | Default | Effect |
| --- | --- | --- |
| `--listen` | `127.0.0.1:7400` | Address to accept connections on (`[ipv6]:port` works; port 0 picks one). |
| `--device` | KORA default | Use this device index. |
| `--cache-mb` | `1024` | Upper bound on buffer bytes cached for the client; least-recently-used buffers are evicted. |
| `--name` | the device name | Name reported to clients. |

It prints `kora-worker listening on port N` and serves until interrupted (Ctrl-C). Connected clients
re-run outstanding work locally when it stops.

## Security

- Set `KORA_WORKER_TOKEN` (the same value on the client) to accept authenticated clients from other
  machines. Without a token, only clients on the same machine are accepted.
- The handshake is an HMAC-SHA256 challenge; every frame carries a CRC32C. Traffic is **not
  encrypted**: run workers on a trusted link or inside an SSH tunnel or VPN.
- One client is served at a time. A newly authenticated client replaces an idle or vanished one, so a
  host that reconnects after a link drop is never locked out; unauthenticated peers cannot displace
  it.

## Protocol

Protocol version 2 (KORA 1.0). Each frame is a 32-byte little-endian header (magic `KORA`, version,
type, request id, payload size, CRC32C) followed by the payload. Peers of different versions reject
each other at the first frame. Workers cache input buffers by version, so unchanged inputs are not
resent; the client releases cached buffers when you drop them.

## Embedding

`kora::WorkerServer::start(WorkerOptions { .. })` runs the same server inside your own process;
`port()`, `stats()`, and `stop()` control it.
