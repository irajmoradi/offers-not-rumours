import json, adaption_backend as adb
lines = ['event: x', 'data: ' + json.dumps({"type": "response.output_text.delta", "delta": '{"action": '}),
         'data: ' + json.dumps({"type": "response.output_text.delta", "delta": '"idle"}'}),
         'data: ' + json.dumps({"type": "response.completed", "response": {"status": "completed", "output": [], "usage": {"cost": 0.0001, "input_tokens": 10, "output_tokens": 5}}}),
         'data: [DONE]']
print(adb.AdaptionBackend.parse_sse(lines))
lines2 = ['data: ' + json.dumps({"type": "response.completed", "response": {"output_text": "hi", "usage": {"cost": 0.1}}})]
print(adb.AdaptionBackend.parse_sse(lines2))
try:
    adb.AdaptionBackend.parse_sse(['data: ' + json.dumps({"type": "response.output_text.delta", "delta": "x"})])
except RuntimeError as e: print("cut ok:", e)
import os; os.environ["ADAPTION_KEY_FILE"] = "/nonexistent"
try: adb.AdaptionBackend("m")
except SystemExit as e: print("keyfile check ok:", str(e)[:60])
