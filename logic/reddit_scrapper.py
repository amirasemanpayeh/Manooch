class RedditScrapper:
    """Reddit scraper with proxy rotation and comprehensive post/comment extraction"""
    subs_to_scrape = ['CasualUK']

    def __init__(self, proxies: list = None):
        """
        Initialize Reddit scrapper with proxy rotation system
        Args:
            proxies: Optional list of proxy URLs to use instead of auto-fetched ones
        """
        from logic.proxy_switcher import ProxySwitcher
        self.proxy_switcher = ProxySwitcher(proxies)

    def scrape(self) -> dict:
        """
        Scrape all subreddits in subs_to_scrape list
        Returns:
            Dictionary with subreddit names as keys and post lists as values
        """
        all_results = {}
        # Iterate through each subreddit in our target list
        for sub in self.subs_to_scrape:
            results = self._scrape_subreddit(sub, sortBy="top", limit=10)
            all_results[sub] = results
        return all_results

    def _get_page_content_json(self, url: str) -> dict:
        """
        Fetch JSON content from URL using proxy rotation
        Args:
            url: The URL to fetch JSON data from
        Returns:
            Parsed JSON data as dictionary, or None if all proxies fail
        """
        import requests
        import time
        tried_proxies = set()
        prefer_last_successful = True
        
        # Keep trying until we find a working proxy or exhaust all proxies
        while len(tried_proxies) < len(self.proxy_switcher.proxies):
            try:
                # Get random user agent for stealth
                user_agent = self.proxy_switcher.get_random_user_agent()
                headers = {'User-Agent': user_agent}
                
                # Get proxy dictionary for requests library
                proxies = self.proxy_switcher.get_requests_proxy_dict(prefer_last_successful=prefer_last_successful)
                proxy_addr = proxies['http']
                
                # Skip if we already tried this proxy in current session
                if proxy_addr in tried_proxies:
                    prefer_last_successful = False
                    continue
                    
                tried_proxies.add(proxy_addr)
                print(f"Trying proxy: {proxy_addr}")
                
                # Make the HTTP request through proxy
                response = requests.get(url, headers=headers, proxies=proxies, timeout=15)
                if response.status_code == 200:
                    print(f"Success with proxy: {proxy_addr}")
                    # Mark this proxy as successful for future use
                    self.proxy_switcher.mark_proxy_successful(proxy_addr)
                    return response.json()
                else:
                    print(f"Proxy {proxy_addr} failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"Proxy {proxy_addr} error: {e}")
                
            # After first attempt, don't prefer last successful proxy
            prefer_last_successful = False
            time.sleep(1)  # Brief delay between proxy attempts
            
        print("All available proxies failed.")
        return None


    def _get_posts_in_subreddit(self, json_data: dict) -> list:
        """
        Extract post data from Reddit API JSON response
        Args:
            json_data: Raw JSON response from Reddit API
        Returns:
            List of raw post data dictionaries
        """
        posts = []
        if json_data:
            # Navigate Reddit's JSON structure: data -> children -> post data
            children = json_data.get("data", {}).get("children", [])
            for child in children:
                post_data = child.get("data", {})
                posts.append(post_data)
        return posts
    

    def _scrape_subreddit(self, subreddit: str, sortBy: str = "top", limit: int = 10) -> list:
        """
        Scrape posts from a specific subreddit
        Args:
            subreddit: Name of the subreddit (without r/)
            sortBy: Sort method (top, hot, new, rising)
            limit: Maximum number of posts to retrieve
        Returns:
            List of parsed post dictionaries
        """
        if not subreddit:
            return []
            
        # Construct Reddit JSON API URL
        url = f"https://www.reddit.com/r/{subreddit}/{sortBy}.json?limit={limit}"
        
        # Fetch raw JSON data using proxy rotation
        data = self._get_page_content_json(url)
        if data:
            # Extract raw posts from JSON response
            posts = self._get_posts_in_subreddit(data)
            results = []
            
            # Parse each post into our standardized format
            for post in posts:
                parsed = self._parse_post(post)
                if parsed:
                    results.append(parsed)
            return results
        else:
            print("Failed to retrieve data")
            return []
        
    def _parse_post(self, post: dict) -> dict:
        """
        Parse raw Reddit post data into standardized format
        Args:
            post: Raw post data dictionary from Reddit API
        Returns:
            Standardized post dictionary with key fields
        """
        if not post:
            return None
            
        # Extract basic post information
        title = post.get("title", "")
        description = post.get("selftext", "")  # Post text content
        has_media = self._has_media(post)
        upvotes = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        
        # Construct full Reddit post URL using permalink
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

    def _has_media(self, post: dict) -> bool:
        """
        Check if a Reddit post contains media (images/videos)
        Args:
            post: Raw post data dictionary from Reddit API
        Returns:
            True if post contains media, False otherwise
        """
        # Check post hint for explicit media type
        if post.get("post_hint") in ["image", "video"]:
            return True
            
        # Check if post is marked as video
        if post.get("is_video"):
            return True
            
        # Check if post has preview images
        if post.get("preview") and post["preview"].get("images"):
            return True
            
        # Check URL for common image/video file extensions
        url = post.get("url", "")
        if url and (url.endswith(".jpg") or url.endswith(".png") or url.endswith(".gif")):
            return True
            
        return False

    def get_post_with_comments(self, post_url: str, top_n_comments: int = 10) -> dict:
        """
        Get full post content and top n comments from a Reddit post URL
        Args:
            post_url: Reddit post URL (e.g. https://www.reddit.com/r/CasualUK/comments/xyz/title/)
            top_n_comments: Number of top comments to retrieve
        Returns:
            Dictionary with post content and comments, or None if failed
        """
        import re
        
        # Extract post ID from URL using regex pattern
        match = re.search(r'/comments/([a-zA-Z0-9]+)/', post_url)
        if not match:
            print(f"Could not extract post ID from URL: {post_url}")
            return None
            
        post_id = match.group(1)
        # Construct JSON API URL for post and comments
        json_url = f"{post_url.rstrip('/')}.json?limit={top_n_comments}&sort=top"
        
        print(f"Fetching post and comments from: {json_url}")
        data = self._get_page_content_json(json_url)
        
        # Reddit returns array: [post_data, comments_data]
        if not data or len(data) < 2:
            print("Failed to retrieve post and comments data")
            return None
            
        # Parse post data (first element in response array)
        post_data = data[0]["data"]["children"][0]["data"]
        
        # Parse comments data (second element in response array)
        comments_data = data[1]["data"]["children"]
        
        # Extract comprehensive post content
        post_content = {
            "title": post_data.get("title", ""),
            "selftext": post_data.get("selftext", ""),  # Post body text
            "author": post_data.get("author", ""),
            "score": post_data.get("score", 0),  # Upvotes minus downvotes
            "num_comments": post_data.get("num_comments", 0),
            "created_utc": post_data.get("created_utc", 0),  # Unix timestamp
            "url": post_data.get("url", ""),  # External URL if any
            "permalink": f"https://www.reddit.com{post_data.get('permalink', '')}",
            "has_media": self._has_media(post_data)
        }
        
        # Extract top comments with full metadata
        comments = []
        for comment_data in comments_data[:top_n_comments]:
            comment = comment_data.get("data", {})
            # Skip deleted, removed, or empty comments
            if comment.get("body") and comment.get("body") not in ["[deleted]", "[removed]"]:
                comments.append({
                    "author": comment.get("author", ""),
                    "body": comment.get("body", ""),  # Comment text content
                    "score": comment.get("score", 0),  # Comment upvotes
                    "created_utc": comment.get("created_utc", 0),
                    "is_submitter": comment.get("is_submitter", False),  # Is original poster
                    "permalink": f"https://www.reddit.com{comment.get('permalink', '')}"
                })
        
        return {
            "post": post_content,
            "comments": comments
        }
