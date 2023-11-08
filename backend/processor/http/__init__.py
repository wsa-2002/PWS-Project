import fastapi


def register_routers(app: fastapi.FastAPI):
    from . import (
        public,
        line,
    )

    app.include_router(public.router)
    app.include_router(line.router)
