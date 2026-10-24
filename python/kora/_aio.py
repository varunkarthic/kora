"""asyncio support for kora.Event (used by Event.__await__)."""

import asyncio


def _resolve(future, error):
    if future.done():
        return
    if error is None:
        future.set_result(None)
    else:
        future.set_exception(error)


async def _wait(event):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    event._register_awaitable(loop, future, _resolve)
    await future


def await_event(event):
    return _wait(event).__await__()
