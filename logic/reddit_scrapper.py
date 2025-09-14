class RedditScrapper:
    subs_to_scrape = ['CasualUK']

    def __init__(self, proxies=None):
        from logic.proxy_switcher import ProxySwitcher
        self.proxy_switcher = ProxySwitcher(proxies)

    def scrape(self):
        all_results = {}
        for sub in self.subs_to_scrape:
            results = self._scrape_subreddit(sub, sortBy="top", limit=10)
            all_results[sub] = results
        return all_results

    def _get_page_content_json(self, url):
        import requests
        import time
        tried_proxies = set()
        prefer_last_successful = True
        
        # Keep trying until we find a working proxy
        while len(tried_proxies) < len(self.proxy_switcher.proxies):
            try:
                user_agent = self.proxy_switcher.get_random_user_agent()
                headers = {'User-Agent': user_agent}
                proxies = self.proxy_switcher.get_requests_proxy_dict(prefer_last_successful=prefer_last_successful)
                proxy_addr = proxies['http']
                
                # Skip if we already tried this proxy
                if proxy_addr in tried_proxies:
                    prefer_last_successful = False
                    continue
                    
                tried_proxies.add(proxy_addr)
                print(f"Trying proxy: {proxy_addr}")
                
                response = requests.get(url, headers=headers, proxies=proxies, timeout=15)
                if response.status_code == 200:
                    print(f"Success with proxy: {proxy_addr}")
                    self.proxy_switcher.mark_proxy_successful(proxy_addr)
                    return response.json()
                else:
                    print(f"Proxy {proxy_addr} failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Proxy {proxy_addr} error: {e}")
                
            prefer_last_successful = False
            time.sleep(1)
            
        print("All available proxies failed.")
        return None


    def _get_posts_in_subreddit(self, json_data: dict):
        posts = []
        if json_data:
            children = json_data.get("data", {}).get("children", [])
            for child in children:
                post_data = child.get("data", {})
                posts.append(post_data)
        return posts
    

    def _scrape_subreddit(self, subreddit: str, sortBy: str = "top", limit=10):
        if not subreddit:
            return
        url = f"https://www.reddit.com/r/{subreddit}/{sortBy}.json?limit={limit}"
        data = self._get_page_content_json(url)
        if data:
            posts = self._get_posts_in_subreddit(data)
            results = []
            for post in posts:
                parsed = self._parse_post(post)
                if parsed:
                    results.append(parsed)
            return results
        else:
            print("Failed to retrieve data")
            return []
        
    def _parse_post(self, post):
        if not post:
            return None
        title = post.get("title", "")
        description = post.get("selftext", "")
        has_media = self._has_media(post)
        upvotes = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        # Get the Reddit post URL using permalink
        permalink = post.get("permalink", "")
        post_url = f"https://www.reddit.com{permalink}" if permalink else ""
        return {
            "title": title,
            "description": description,
            "has_media": has_media,
            "upvotes": upvotes,
            "num_comments": num_comments,
            "url": post_url
        }

    def _has_media(self, post):
        # Check for media presence
        if post.get("post_hint") in ["image", "video"]:
            return True
        if post.get("is_video"):
            return True
        if post.get("preview") and post["preview"].get("images"):
            return True
        if post.get("url") and (post["url"].endswith(".jpg") or post["url"].endswith(".png") or post["url"].endswith(".gif")):
            return True
        return False
