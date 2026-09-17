"""
Re-export services.insurance_api.main for hyphenated folder compatibility.
"""
from services.insurance_api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
