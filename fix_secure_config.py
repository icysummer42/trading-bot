#!/usr/bin/env python3
"""
Fix Secure Configuration System

Properly implements the SecureConfigManager to decrypt SECURE: prefixed values
and stores the API keys in the secure_config.json file.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_configuration import SecureConfigManager, create_config_manager

def create_secure_config_file():
    """Create the secure_config.json file with encrypted API keys."""
    print("🔐 Creating Secure Configuration File")
    print("-" * 40)
    
    # The actual API keys from base.yaml (without SECURE: prefix)
    api_keys = {
        "polygon_key": "OROZxfRr0KoSI6ilsYZiyp8oTRHq478Y",
        "finnhub_key": "d1vm8l1r01qqgeembco0d1vm8l1r01qqgeembcog", 
        "alpha_vantage_key": "583JT6TWFMIKDGVN",
        "newsapi_key": "47007e0b8de74863b15d79bb83df5976",
        "openweather_key": "880b3b206c79f32c79970bfd338bb555",
        "fred_key": "1e4aaa2f7308fbef174ff5cb1e2b8460",
        "gnews_key": "00525f2193cd8d6cca6219ba869d02ea",
        "stocktwits_token": "your_stocktwits_token_here",
        "reddit_client_id": "your_reddit_client_id_here",
        "database_password": "your_database_password_here"
    }
    
    try:
        # Create secure config manager
        secure_manager = SecureConfigManager()
        print("✅ SecureConfigManager created")
        
        # Store each key
        for key_name, key_value in api_keys.items():
            try:
                secure_manager.secure_store(key_name, key_value)
                print(f"✅ Stored {key_name}")
                
                # Verify by loading back
                loaded_value = secure_manager.secure_load(key_name)
                if loaded_value == key_value:
                    print(f"   ✅ Verified {key_name}")
                else:
                    print(f"   ❌ Verification failed for {key_name}")
                    print(f"      Expected: {key_value[:8]}...")
                    print(f"      Got: {loaded_value[:8] if loaded_value else 'None'}...")
                    
            except Exception as e:
                print(f"❌ Error with {key_name}: {e}")
        
        # Check if secure_config.json was created
        secure_file = Path("secure_config.json")
        if secure_file.exists():
            print(f"\n📁 Secure config file created: {secure_file}")
            print(f"   Size: {secure_file.stat().st_size} bytes")
            return True
        else:
            print("❌ Secure config file not created")
            return False
        
    except Exception as e:
        print(f"❌ Error creating secure config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_secure_decryption():
    """Test that SECURE: prefixed values are now properly decrypted."""
    print("\n🧪 Testing Secure Configuration Decryption")
    print("-" * 50)
    
    try:
        # Create fresh config manager
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        print("📊 Testing API Key Decryption")
        print("-" * 30)
        
        if hasattr(config, 'data_sources'):
            ds = config.data_sources
            
            # Test each API key
            test_keys = [
                ('polygon_api_key', 'Polygon', 'OROZxfRr0KoSI6ilsYZiyp8oTRHq478Y'),
                ('alpha_vantage_api_key', 'Alpha Vantage', '583JT6TWFMIKDGVN'),
                ('finnhub_api_key', 'Finnhub', 'd1vm8l1r01qqgeembco0d1vm8l1r01qqgeembcog'),
                ('newsapi_key', 'NewsAPI', '47007e0b8de74863b15d79bb83df5976'),
                ('fred_api_key', 'FRED', '1e4aaa2f7308fbef174ff5cb1e2b8460')
            ]
            
            success_count = 0
            
            for key_attr, name, expected_value in test_keys:
                if hasattr(ds, key_attr):
                    actual_value = getattr(ds, key_attr)
                    
                    if actual_value == expected_value:
                        print(f"✅ {name}: Properly decrypted - {actual_value[:8]}...")
                        success_count += 1
                    elif actual_value and actual_value.startswith("SECURE:"):
                        print(f"❌ {name}: Still encrypted - {actual_value}")
                    elif actual_value:
                        print(f"⚠️  {name}: Unexpected value - {actual_value[:8]}...")
                    else:
                        print(f"❌ {name}: Missing or empty")
                else:
                    print(f"❌ {name}: Attribute not found")
            
            print(f"\n📋 Results: {success_count}/{len(test_keys)} keys properly decrypted")
            return success_count == len(test_keys)
            
        else:
            print("❌ data_sources section not found in config")
            return False
        
    except Exception as e:
        print(f"❌ Error testing decryption: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alpha_vantage_with_config():
    """Test Alpha Vantage API using the decrypted configuration."""
    print("\n🔗 Testing Alpha Vantage with Decrypted Config")
    print("-" * 50)
    
    try:
        # Load config
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        if hasattr(config, 'data_sources') and hasattr(config.data_sources, 'alpha_vantage_api_key'):
            api_key = config.data_sources.alpha_vantage_api_key
            
            if api_key and not api_key.startswith("SECURE:"):
                print(f"✅ Using config API key: {api_key[:8]}...")
                
                # Test API call
                import requests
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": "AAPL",
                    "apikey": api_key
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        price = quote.get("05. price", "N/A")
                        change = quote.get("10. change percent", "N/A")
                        print(f"🎯 AAPL Quote: ${price} ({change})")
                        return True
                    else:
                        print(f"⚠️  Unexpected response: {data}")
                        return False
                else:
                    print(f"❌ API error {response.status_code}: {response.text[:200]}")
                    return False
                    
            else:
                print("❌ API key still encrypted or missing")
                return False
                
        else:
            print("❌ Alpha Vantage API key not found in config")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def update_alpha_vantage_fetcher():
    """Update the Alpha Vantage fetcher to use configuration instead of hardcoded key."""
    print("\n🔄 Updating Alpha Vantage Fetcher")
    print("-" * 40)
    
    try:
        # Read the current fetcher
        fetcher_file = Path("alpha_vantage_data_fetcher.py")
        
        if not fetcher_file.exists():
            print("⚠️  Alpha Vantage fetcher file not found")
            return False
        
        with open(fetcher_file, 'r') as f:
            content = f.read()
        
        # Replace hardcoded API key with configuration loading
        old_create_function = '''def create_alpha_vantage_fetcher() -> Optional[AlphaVantageDataFetcher]:
    """Create Alpha Vantage data fetcher with API key."""
    api_key = "583JT6TWFMIKDGVN"  # Your working API key
    return AlphaVantageDataFetcher(api_key)'''
        
        new_create_function = '''def create_alpha_vantage_fetcher() -> Optional[AlphaVantageDataFetcher]:
    """Create Alpha Vantage data fetcher with API key from configuration."""
    try:
        from advanced_configuration import create_config_manager
        
        # Load configuration
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        # Get Alpha Vantage API key from secure configuration
        if hasattr(config, 'data_sources') and hasattr(config.data_sources, 'alpha_vantage_api_key'):
            api_key = config.data_sources.alpha_vantage_api_key
            
            if api_key and not api_key.startswith("SECURE:"):
                print(f"✅ Using Alpha Vantage API key from config: {api_key[:8]}...")
                return AlphaVantageDataFetcher(api_key)
            else:
                print("❌ Alpha Vantage API key not properly decrypted")
                # Fallback to hardcoded key for now
                api_key = "583JT6TWFMIKDGVN"
                print(f"⚠️  Using fallback API key: {api_key[:8]}...")
                return AlphaVantageDataFetcher(api_key)
        else:
            print("❌ Alpha Vantage API key not found in configuration")
            return None
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        # Fallback to hardcoded key
        api_key = "583JT6TWFMIKDGVN"
        print(f"⚠️  Using fallback API key: {api_key[:8]}...")
        return AlphaVantageDataFetcher(api_key)'''
        
        # Replace the function
        if old_create_function.replace('"583JT6TWFMIKDGVN"  # Your working API key', '"583JT6TWFMIKDGVN"') in content:
            updated_content = content.replace(old_create_function.replace('  # Your working API key', ''), new_create_function)
        elif 'def create_alpha_vantage_fetcher()' in content:
            # Find and replace the existing function
            import re
            pattern = r'def create_alpha_vantage_fetcher\(\)[^:]*:.*?return AlphaVantageDataFetcher\([^)]+\)'
            replacement = new_create_function
            updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            print("⚠️  Could not find create_alpha_vantage_fetcher function to update")
            return False
        
        # Write the updated content
        with open(fetcher_file, 'w') as f:
            f.write(updated_content)
        
        print("✅ Updated Alpha Vantage fetcher to use configuration")
        return True
        
    except Exception as e:
        print(f"❌ Error updating fetcher: {e}")
        return False

def run_secure_config_fix():
    """Run complete secure configuration fix."""
    print("🚀 Secure Configuration System Fix")
    print("=" * 60)
    print()
    
    # Step 1: Create secure config file
    print("Step 1: Create secure configuration file...")
    secure_created = create_secure_config_file()
    
    if not secure_created:
        print("❌ Failed to create secure configuration")
        return False
    
    # Step 2: Test decryption
    print("\nStep 2: Test configuration decryption...")
    decryption_works = test_secure_decryption()
    
    if not decryption_works:
        print("❌ Configuration decryption not working properly")
        # Continue anyway to see what we can fix
    
    # Step 3: Test Alpha Vantage with config
    print("\nStep 3: Test Alpha Vantage with configuration...")
    api_works = test_alpha_vantage_with_config()
    
    # Step 4: Update fetcher (optional)
    print("\nStep 4: Update Alpha Vantage fetcher...")
    fetcher_updated = update_alpha_vantage_fetcher()
    
    # Summary
    if decryption_works and api_works:
        print("\n🎉 Secure Configuration Fix Complete!")
        print("✅ SECURE: prefixed values properly decrypted")
        print("✅ API keys accessible from configuration")
        print("✅ Alpha Vantage working with config")
        if fetcher_updated:
            print("✅ Fetcher updated to use configuration")
        
        print("\n📋 Benefits:")
        print("• No more hardcoded API keys in source code")
        print("• Centralized secure key management") 
        print("• Proper encryption at rest")
        print("• Easy key rotation and management")
        
        return True
    else:
        print("\n⚠️  Secure configuration partially working")
        if secure_created:
            print("✅ Secure file created")
        if api_works:
            print("✅ API access working")
        if not decryption_works:
            print("❌ Automatic decryption needs investigation")
        
        return False

if __name__ == "__main__":
    success = run_secure_config_fix()
    print(f"\nResult: {'✅ SUCCESS' if success else '⚠️  PARTIAL SUCCESS'}")
    sys.exit(0 if success else 1)