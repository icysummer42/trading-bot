#!/usr/bin/env python3
"""
Debug Secure Configuration System

Investigates why SECURE: prefixed values are not being decrypted automatically
and fixes the issue to properly load encrypted API keys.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_configuration import create_config_manager, SecureConfigManager

def debug_secure_manager():
    """Debug the SecureConfigManager to see what's happening."""
    print("🔍 Debugging Secure Configuration Manager")
    print("-" * 50)
    
    try:
        # Create secure manager directly
        secure_manager = SecureConfigManager()
        print("✅ SecureConfigManager created")
        
        # Check if secure directory exists
        secure_dir = Path(secure_manager.secure_config_dir)
        print(f"📁 Secure directory: {secure_dir}")
        print(f"   Exists: {secure_dir.exists()}")
        
        if secure_dir.exists():
            secure_files = list(secure_dir.glob("*"))
            print(f"   Files: {[f.name for f in secure_files]}")
            
            # Try to list expected keys
            expected_keys = [
                "polygon_key",
                "finnhub_key", 
                "alpha_vantage_key",
                "newsapi_key",
                "fred_key"
            ]
            
            for key in expected_keys:
                key_file = secure_dir / f"{key}.enc"
                print(f"   {key}: {'✅' if key_file.exists() else '❌'} {key_file}")
        else:
            print("❌ Secure directory doesn't exist!")
            
        # Test loading a key
        print("\n🔑 Testing Key Loading")
        print("-" * 30)
        
        test_key = "polygon_key"
        try:
            value = secure_manager.secure_load(test_key)
            if value:
                print(f"✅ {test_key} loaded: {value[:8]}...")
            else:
                print(f"⚠️  {test_key} returned None/empty")
        except Exception as e:
            print(f"❌ Error loading {test_key}: {e}")
            
        return secure_manager
        
    except Exception as e:
        print(f"❌ Error creating SecureConfigManager: {e}")
        return None

def debug_config_loading():
    """Debug the full configuration loading process."""
    print("\n🔧 Debugging Configuration Loading")
    print("-" * 50)
    
    try:
        # Create config manager
        config_manager = create_config_manager(environment="development")
        
        # Get raw configuration before processing
        print("📋 Raw Configuration Check")
        print("-" * 30)
        
        # Check if the raw YAML still has SECURE: values
        config_dir = Path("config")
        base_file = config_dir / "base.yaml"
        
        if base_file.exists():
            with open(base_file, 'r') as f:
                content = f.read()
            
            # Check for SECURE: references
            secure_refs = [line.strip() for line in content.split('\n') if 'SECURE:' in line]
            print("SECURE: references in base.yaml:")
            for ref in secure_refs[:5]:  # Show first 5
                print(f"   {ref}")
        
        # Get processed configuration
        config = config_manager.get_config()
        
        print("\n📊 Processed Configuration Check")
        print("-" * 30)
        
        if hasattr(config, 'data_sources'):
            ds = config.data_sources
            
            keys_to_check = [
                ('polygon_api_key', 'Polygon'),
                ('alpha_vantage_api_key', 'Alpha Vantage'),
                ('finnhub_api_key', 'Finnhub')
            ]
            
            for key_attr, name in keys_to_check:
                if hasattr(ds, key_attr):
                    value = getattr(ds, key_attr)
                    if value:
                        if value.startswith("SECURE:"):
                            print(f"❌ {name}: Still has SECURE: prefix - not decrypted")
                            print(f"   Raw value: {value}")
                        else:
                            print(f"✅ {name}: Decrypted - {value[:8]}..." if len(value) > 8 else f"✅ {name}: {value}")
                    else:
                        print(f"⚠️  {name}: Empty/None")
                else:
                    print(f"❌ {name}: Attribute missing")
        
        return config_manager
        
    except Exception as e:
        print(f"❌ Error in configuration loading: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_secure_keys():
    """Create encrypted secure keys from the plaintext values."""
    print("\n🔐 Creating Secure Keys")
    print("-" * 30)
    
    # The actual API keys from base.yaml (removing SECURE: prefix)
    api_keys = {
        "polygon_key": "OROZxfRr0KoSI6ilsYZiyp8oTRHq478Y",
        "finnhub_key": "d1vm8l1r01qqgeembco0d1vm8l1r01qqgeembcog", 
        "alpha_vantage_key": "583JT6TWFMIKDGVN",
        "newsapi_key": "47007e0b8de74863b15d79bb83df5976",
        "openweather_key": "880b3b206c79f32c79970bfd338bb555",
        "fred_key": "1e4aaa2f7308fbef174ff5cb1e2b8460",
        "gnews_key": "00525f2193cd8d6cca6219ba869d02ea"
    }
    
    try:
        # Create secure manager
        secure_manager = SecureConfigManager()
        
        # Ensure secure directory exists
        secure_dir = Path(secure_manager.secure_config_dir)
        secure_dir.mkdir(exist_ok=True)
        print(f"📁 Secure directory: {secure_dir}")
        
        # Save each key
        for key_name, key_value in api_keys.items():
            try:
                secure_manager.secure_save(key_name, key_value)
                print(f"✅ Saved {key_name}")
                
                # Verify by loading back
                loaded_value = secure_manager.secure_load(key_name)
                if loaded_value == key_value:
                    print(f"   ✅ Verified {key_name}")
                else:
                    print(f"   ❌ Verification failed for {key_name}")
                    
            except Exception as e:
                print(f"❌ Error saving {key_name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating secure keys: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_configuration():
    """Test if the configuration now properly decrypts SECURE: values."""
    print("\n🧪 Testing Fixed Configuration")
    print("-" * 40)
    
    try:
        # Create fresh config manager
        config_manager = create_config_manager(environment="development")
        config = config_manager.get_config()
        
        if hasattr(config, 'data_sources'):
            ds = config.data_sources
            
            # Test key APIs
            test_results = []
            
            # Alpha Vantage (we know this works)
            if hasattr(ds, 'alpha_vantage_api_key'):
                av_key = ds.alpha_vantage_api_key
                if av_key and not av_key.startswith("SECURE:"):
                    print(f"✅ Alpha Vantage key decrypted: {av_key[:8]}...")
                    
                    # Quick API test
                    import requests
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": "GLOBAL_QUOTE",
                        "symbol": "AAPL",
                        "apikey": av_key
                    }
                    
                    try:
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if "Global Quote" in data:
                                price = data["Global Quote"].get("05. price", "N/A")
                                print(f"   🎯 API Test: AAPL = ${price}")
                                test_results.append(("Alpha Vantage", True))
                            else:
                                print(f"   ⚠️  API returned: {data}")
                                test_results.append(("Alpha Vantage", False))
                    except Exception as e:
                        print(f"   ❌ API test failed: {e}")
                        test_results.append(("Alpha Vantage", False))
                else:
                    print("❌ Alpha Vantage key still encrypted or missing")
                    test_results.append(("Alpha Vantage", False))
            
            # Show other keys
            other_keys = [
                ('polygon_api_key', 'Polygon'),
                ('finnhub_api_key', 'Finnhub'),
                ('newsapi_key', 'NewsAPI'),
                ('fred_api_key', 'FRED')
            ]
            
            for key_attr, name in other_keys:
                if hasattr(ds, key_attr):
                    key_value = getattr(ds, key_attr)
                    if key_value and not key_value.startswith("SECURE:"):
                        print(f"✅ {name} key decrypted: {key_value[:8]}...")
                        test_results.append((name, True))
                    else:
                        print(f"❌ {name} key still encrypted or missing")
                        test_results.append((name, False))
            
            # Summary
            successful = sum(1 for _, success in test_results if success)
            total = len(test_results)
            
            print(f"\n📊 Results: {successful}/{total} keys properly decrypted")
            
            return successful == total
        
        return False
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def run_secure_config_debug():
    """Run complete secure configuration debugging and fixing."""
    print("🚀 Secure Configuration Debug & Fix")
    print("=" * 60)
    print(f"Started: {os.path.basename(__file__)}")
    print()
    
    # Step 1: Debug secure manager
    secure_manager = debug_secure_manager()
    
    # Step 2: Debug config loading
    config_manager = debug_config_loading()
    
    # Step 3: Create secure keys if needed
    print("\nStep 3: Check if we need to create secure keys...")
    if not secure_manager or not Path(secure_manager.secure_config_dir).exists():
        print("🔐 Creating secure keys...")
        keys_created = create_secure_keys()
        if not keys_created:
            print("❌ Failed to create secure keys")
            return False
    else:
        print("✅ Secure keys directory exists")
    
    # Step 4: Test fixed configuration
    print("\nStep 4: Test configuration decryption...")
    config_works = test_fixed_configuration()
    
    if config_works:
        print("\n🎉 Secure Configuration Fix Complete!")
        print("✅ SECURE: values are now properly decrypted")
        print("✅ API keys available to pipeline")
        print("✅ No more hardcoded keys needed")
        
        print("\n📋 Next Steps:")
        print("• Update Alpha Vantage fetcher to use config.data_sources.alpha_vantage_api_key")
        print("• Remove hardcoded API keys from code")
        print("• Test full pipeline with decrypted keys")
        
        return True
    else:
        print("\n⚠️  Secure configuration still has issues")
        print("• Some keys may not be decrypting properly")
        print("• Check encryption/decryption process") 
        return False

if __name__ == "__main__":
    success = run_secure_config_debug()
    print(f"\nResult: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    sys.exit(0 if success else 1)