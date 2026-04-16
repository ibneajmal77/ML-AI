# app/main.py — add this line to register the router
from app.routers import tickets
app.include_router(tickets.router)