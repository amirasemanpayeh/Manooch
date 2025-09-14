class RedditScrapper:
    subs_to_scrape = ['CasualUK']

    def __init__(self):
        pass

    def scrape(self):
        all_results = {}
        for sub in self.subs_to_scrape:
            results = self._scrape_subreddit(sub, sortBy="top", limit=10)
            all_results[sub] = results
        return all_results

    def _get_page_content_json(self, url):
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
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
