import random
import requests
import time
from threading import Lock

class ProxySwitcher:
	USER_AGENTS = [
		# Chrome, Firefox, Edge, Safari, mobile, etc.
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
		"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
		"Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
		"Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Mobile Safari/537.36"
	]
	def get_random_user_agent(self):
		return random.choice(self.USER_AGENTS)
	FREE_PROXY_SOURCES = [
		"https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
		"https://www.proxyscrape.com/api?request=get&format=textplain&protocol=http&timeout=10000&country=all",
		"https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all&format=textplain",
		"https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
		"https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
		"https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
		"https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",
		"https://free-proxy-list.net/",
		"https://www.proxy-list.download/api/v1/get?type=http"
	]

	def __init__(self, proxies=None, auto_refresh=True, refresh_interval=1800, storage_path="storage/proxy_storage.json", test_url="https://httpbin.org/ip", shortlist_size=100, test_limit=1000):
		"""
		proxies: list of proxy URLs, e.g. ["http://ip1:port", "http://ip2:port"]
		auto_refresh: whether to periodically refresh proxies from free sources
		refresh_interval: seconds between refreshes
		"""
		self.lock = Lock()
		self.proxies = proxies or []
		self.last_refresh = 0
		self.refresh_interval = refresh_interval
		self.auto_refresh = auto_refresh
		self.storage_path = storage_path
		self.last_successful_proxy = self._load_last_successful_proxy()
		self.test_url = test_url
		self.shortlist_size = shortlist_size
		self.test_limit = test_limit
		if auto_refresh:
			self.refresh_proxies()
		print(f"Fetched {len(self.proxies)} proxies from sources.")
	def _shortlist_proxies(self):
		import requests
		import time
		from concurrent.futures import ThreadPoolExecutor, as_completed
		print(f"Testing up to {self.test_limit} proxies to shortlist {self.shortlist_size} working ones...")
		
		def test_proxy(proxy):
			try:
				proxy_dict = {"http": proxy, "https": proxy}
				start = time.time()
				resp = requests.head(self.test_url, proxies=proxy_dict, timeout=2)
				latency = time.time() - start
				if resp.ok and latency < 3:
					return (proxy, latency)
			except Exception:
				pass
			return None

		working = []
		test_proxies = self.proxies[:self.test_limit]
		
		print(f"Starting concurrent proxy testing...")
		with ThreadPoolExecutor(max_workers=50) as executor:
			futures = {executor.submit(test_proxy, proxy): proxy for proxy in test_proxies}
			tested = 0
			for future in as_completed(futures, timeout=60):
				tested += 1
				result = future.result()
				if result:
					working.append(result)
					print(f"Proxy {result[0]} OK ({result[1]:.2f}s) - Found {len(working)} working")
					if len(working) >= self.shortlist_size:
						break
				if tested % 100 == 0:
					print(f"Tested {tested}/{len(test_proxies)} proxies...")
		
		if not working:
			print("No working proxies found. Will use direct connection.")
			self.proxies = []
		else:
			# Sort by latency, keep fastest
			working.sort(key=lambda x: x[1])
			self.proxies = [p for p, _ in working[:self.shortlist_size]]
			print(f"Shortlisted {len(self.proxies)} working proxies.")
	def _load_last_successful_proxy(self):
		import os, json
		if os.path.exists(self.storage_path):
			try:
				with open(self.storage_path, "r") as f:
					data = json.load(f)
					return data.get("last_successful_proxy")
			except Exception:
				return None
		return None

	def _save_last_successful_proxy(self, proxy):
		import json
		try:
			with open(self.storage_path, "w") as f:
				json.dump({"last_successful_proxy": proxy}, f)
		except Exception as e:
			print(f"Failed to save proxy: {e}")

	def refresh_proxies(self):
		import requests
		with self.lock:
			proxies = set(self.proxies)
			for url in self.FREE_PROXY_SOURCES:
				try:
					print(f"Fetching from {url[:50]}...")
					if url.endswith(".txt") or "proxyscrape.com" in url or "proxy-list.download" in url:
						resp = requests.get(url, timeout=5)
						if resp.ok:
							for line in resp.text.splitlines():
								line = line.strip()
								if line and ":" in line and not line.startswith("#"):
									# Handle different formats: ip:port or just validate ip:port
									if line.count(":") >= 1:
										ip_port = line.split(":")[0] + ":" + line.split(":")[1]
										proxies.add(f"http://{ip_port}")
					elif "free-proxy-list.net" in url:
						resp = requests.get(url, timeout=5)
						if resp.ok:
							from bs4 import BeautifulSoup
							soup = BeautifulSoup(resp.text, "html.parser")
							table = soup.find("table", id="proxylisttable")
							if table and table.tbody:
								for row in table.tbody.find_all("tr"):
									cols = row.find_all("td")
									if len(cols) >= 7:
										ip = cols[0].text.strip()
										port = cols[1].text.strip()
										https = cols[6].text.strip()
										if ip and port and https == "yes":
											proxies.add(f"http://{ip}:{port}")
					print(f"Got {len(proxies)} proxies so far...")
				except Exception as e:
					print(f"Proxy fetch error from {url}: {e}")
			self.proxies = list(proxies)
			self.last_refresh = time.time()

	def get_random_proxy(self, prefer_last_successful=True):
		if self.auto_refresh and (time.time() - self.last_refresh > self.refresh_interval):
			self.refresh_proxies()
		with self.lock:
			# Always try last successful proxy first
			if prefer_last_successful and self.last_successful_proxy:
				return self.last_successful_proxy
			# If no last successful or it failed, get a random one from the full list
			if not self.proxies:
				self.refresh_proxies()
			if not self.proxies:
				raise Exception("No proxies available - cannot proceed without proxy")
			return random.choice(self.proxies)

	def get_requests_proxy_dict(self, prefer_last_successful=True):
		try:
			proxy = self.get_random_proxy(prefer_last_successful=prefer_last_successful)
			return {
				"http": proxy,
				"https": proxy
			}
		except Exception as e:
			print(f"Error getting proxy: {e}")
			raise e

	def mark_proxy_successful(self, proxy):
		self.last_successful_proxy = proxy
		self._save_last_successful_proxy(proxy)
