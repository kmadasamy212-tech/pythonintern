"""
Vercel Serverless Function Entry-Point.
Exposes the FastAPI application to Vercel's Python Serverless Runtime.
"""

import os
import sys

# Add root directory to sys.path so modules resolve cleanly in Vercel runtime
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import the main FastAPI app
from main import app

# Vercel looks for 'app' or 'handler'
handler = app
