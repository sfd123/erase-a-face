#!/usr/bin/env python3
"""
Main entry point for Golf Video Anonymizer service
"""

import uvicorn
from fastapi import FastAPI

# This will be implemented in later tasks
app = FastAPI(
    title="Golf Video Anonymizer",
    description="Service for anonymizing faces in golf swing videos",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Golf Video Anonymizer API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )