def run_reddit_scraping():
    """
    Reddit scraping functionality
    """
    from logic.reddit_scrapper import RedditScrapper
    from utils.settings_manager import settings
    from utils.supabase_manager import SupabaseManager
    from utils.modal_manager import ModalManager
    
    print("🔍 Starting Reddit scraping...")
    
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


def run_prompt_generation():
    """
    Generate prompt from story file functionality
    """
    from logic.shorts_strategy_manager import StrategyManager
    
    print("📝 Generate Prompt from Story File")
    print()
    
    # Get file paths from user
    story_path = input("Enter path to story JSON file: ").strip()
    if not story_path:
        print("❌ Story file path is required")
        return
    
    strategy_path = input("Enter path to strategy JSON file: ").strip()
    if not strategy_path:
        print("❌ Strategy file path is required")
        return
    
    # Validate files exist
    import os
    if not os.path.exists(story_path):
        print(f"❌ Story file not found: {story_path}")
        return
    
    if not os.path.exists(strategy_path):
        print(f"❌ Strategy file not found: {strategy_path}")
        return
    
    try:
        print("🚀 Generating prompt...")
        manager = StrategyManager()
        prompt = manager.generate_prompt_from_story_file(story_path, strategy_path)
        
        print()
        print("=" * 80)
        print("GENERATED PROMPT")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print()
        print("✅ Prompt generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating prompt: {e}")


def display_menu():
    """
    Display the main menu options
    """
    print()
    print("🎬 Manooch - Video Content Generation Tool")
    print("=" * 50)
    print("1. Reddit Scraping")
    print("2. Generate Prompt from Story File")
    print("3. Exit")
    print("=" * 50)


def main() -> None:
    """
    Main application entry point with interactive CLI menu
    """
    while True:
        display_menu()
        
        try:
            choice = input("Select an option (1-3): ").strip()
            
            if choice == "1":
                run_reddit_scraping()
            elif choice == "2":
                run_prompt_generation()
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
        
        # Ask if user wants to continue
        if choice in ["1", "2"]:
            print()
            continue_choice = input("Press Enter to return to menu or 'q' to quit: ").strip().lower()
            if continue_choice == 'q':
                print("👋 Goodbye!")
                break

if __name__ == "__main__":
    main()
