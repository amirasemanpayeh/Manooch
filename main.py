def main() -> None:
    """
    Main application entry point for Reddit scraping
    Demonstrates basic subreddit scraping functionality
    """
    from logic.reddit_scrapper import RedditScrapper
    from utils.settings_manager import settings
    from utils.supabase_manager import SupabaseManager
    from utils.modal_manager import ModalManager
    
    # Initialize core services
    supabase_manager = SupabaseManager(url=settings.supabase_url, key=settings.supabase_key)
    modal_manager = ModalManager()

    # Initialize scrapper with auto-refreshing free proxies
    scrapper = RedditScrapper(proxies=None)
    
    # Scrape all configured subreddits
    all_results = scrapper.scrape()
    
    # Display results in formatted output
    for subreddit, posts in all_results.items():
        print(f"Subreddit: {subreddit}")
        for idx, post in enumerate(posts, 1):
            print(f"  Post {idx}:")
            print(f"    Title: {post['title']}")
            # Truncate description for readability
            description = post['description'][:100] + "..." if len(post['description']) > 100 else post['description']
            print(f"    Description: {description}")
            print(f"    Has Media: {post['has_media']}")
            print(f"    Upvotes: {post['upvotes']}")
            print(f"    Comments: {post['num_comments']}")
            print(f"    URL: {post['url']}")
            print()

if __name__ == "__main__":
    main()
