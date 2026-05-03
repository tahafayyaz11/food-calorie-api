from pyngrok import ngrok
import uvicorn

public_url = ngrok.connect(5000)
print(f"\n{'='*50}")
print(f"🌐 Public URL: {public_url}")
print(f"📱 Use this in App.js as API_URL!")
print(f"{'='*50}\n")

uvicorn.run("main:app", host="0.0.0.0", port=5000)