import ssl, certifi
import urllib.request

print("[DEBUG] Python SSL default paths:", ssl.get_default_verify_paths())
print("[DEBUG] certifi bundle:", certifi.where())

# Test fetching a HTTPS site
url = "https://twitter.com"
try:
    resp = urllib.request.urlopen(url)
    print("[DEBUG] urllib request worked! Code:", resp.status)
except Exception as e:
    print("[ERROR] urllib failed:", e)
