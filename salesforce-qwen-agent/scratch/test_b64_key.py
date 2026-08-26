import base64

key_b64 = "M01WRzk3TDdPV2JQcTZVelRTTzAyUTBZeEdSQ1hMRWljVmtXb0dEQnZtX2trcEpGMlBoeFdSRmpEanZTQnl0NjE4TDk0NmxiQmdUZWpqa3h5Y19IbQ=="
secret_b64 = "OEZBQzMyMUJGMTg3QkY5QUY1NzJGQzMwRTU4MTkzMzAyMDhGMDI1N0FBMjdDMzEyODEwNUM1NUJBRkZEQjBFNg=="

decoded_key = base64.b64decode(key_b64).decode("utf-8")
decoded_secret = base64.b64decode(secret_b64).decode("utf-8")

print("Decoded Key:", decoded_key)
print("Decoded Secret:", decoded_secret)
