def main():

    from logic.reddit_scrapper import RedditScrapper
    # Use ProxySwitcher with auto-refreshing free proxies
    scrapper = RedditScrapper(proxies=None)
    all_results = scrapper.scrape()
    for subreddit, posts in all_results.items():
        print(f"Subreddit: {subreddit}")
        for idx, post in enumerate(posts, 1):
            print(f"  Post {idx}:")
            print(f"    Title: {post['title']}")
            print(f"    Description: {post['description'][:100]}...")
            print(f"    Has Media: {post['has_media']}")
            print(f"    Upvotes: {post['upvotes']}")
            print(f"    Comments: {post['num_comments']}")
            print(f"    URL: {post['url']}")
            print()

if __name__ == "__main__":
    main()
