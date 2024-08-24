#!/bin/bash
cd backend/
fastapi run main.py &
cd ../frontend/
npm run dev