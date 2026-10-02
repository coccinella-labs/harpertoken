# API Serving

After training, serve the model via API:
```sh
pip install fastapi uvicorn
uvicorn scripts.api:app --reload
```

`scripts/api.py` only defines the FastAPI app. Running it directly with `python`
does nothing, because the module has no `__main__` block and never calls
`uvicorn.run`, so the process imports and exits immediately.

## API Endpoints

- `GET /`: API info
- `POST /predict`: QA prediction (json: {"question": "...", "context": "..."})

## Example Usage

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Its capital is Paris."
})
print(response.json())
```
