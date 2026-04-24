#!/usr/bin/env python3
"""
Test script to verify cdsapi installation and Copernicus CDS connection.

This script:
1. Checks if cdsapi is installed
2. Verifies ~/.cdsapirc credentials file exists
3. Tests connection to Copernicus CDS
4. Downloads a small sample of ERA5-Land data (1 day, 1 variable)

Run this before attempting full data downloads to ensure everything is set up correctly.

Usage:
    conda activate climate-risk
    python test_cdsapi_connection.py
"""

import sys
import os
from pathlib import Path


def check_cdsapi_installed():
    """Check if cdsapi package is installed."""
    print("1. Checking if cdsapi is installed...")
    try:
        import cdsapi
        print(f"   ✓ cdsapi is installed (version: {cdsapi.__version__ if hasattr(cdsapi, '__version__') else 'unknown'})")
        return True
    except ImportError:
        print("   ✗ cdsapi is NOT installed")
        print("   → Run: pip install cdsapi")
        return False


def check_credentials_file():
    """Check if ~/.cdsapirc credentials file exists."""
    print("\n2. Checking for CDS API credentials file (~/.cdsapirc)...")
    cdsapirc_path = Path.home() / ".cdsapirc"

    if cdsapirc_path.exists():
        print(f"   ✓ Found: {cdsapirc_path}")
        return True
    else:
        print(f"   ✗ NOT found: {cdsapirc_path}")
        print("\n   → To create credentials file:")
        print("      1. Go to: https://cds.climate.copernicus.eu")
        print("      2. Create a free account (if you don't have one)")
        print("      3. Go to your profile → API key")
        print("      4. Copy your UID and API key")
        print("      5. Create ~/.cdsapirc with these contents:")
        print("\n         url: https://cds.climate.copernicus.eu/api/v2")
        print("         key: YOUR_UID:YOUR_API_KEY")
        print("\n      6. Save and close the file")
        return False


def test_cds_connection():
    """Test connection to Copernicus CDS by retrieving a small data sample."""
    print("\n3. Testing connection to Copernicus CDS...")
    print("   (Downloading 1 day of ERA5-Land data as a test...)")

    try:
        import cdsapi

        # Initialize client
        client = cdsapi.Client()
        print("   ✓ Client initialized successfully")

        # Request a small sample: 1 day, 1 location, 1 variable
        # This should be quick (< 1 minute)
        print("\n   Requesting data from CDS (this may take 1-2 minutes)...")

        request = {
            'dataset': 'reanalysis-era5-land',
            'variable': '2m_temperature',
            'year': '2020',
            'month': '01',
            'day': '01',
            'time': ['12:00'],
            'area': [58, 22, 54, 28],  # Estonia bounding box [N, W, S, E]
            'format': 'netcdf'
        }

        output_file = 'test_era5_sample.nc'

        result = client.retrieve('reanalysis-era5-land', request, output_file)

        # Check if file was created
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / 1024  # Size in KB
            print(f"   ✓ Data downloaded successfully!")
            print(f"   ✓ File: {output_file} ({file_size:.1f} KB)")

            # Try to read with xarray to verify it's valid
            try:
                import xarray as xr
                ds = xr.open_dataset(output_file)
                print(f"   ✓ File is valid NetCDF")
                print(f"   ✓ Dimensions: {dict(ds.dims)}")
                print(f"   ✓ Variables: {list(ds.data_vars.keys())}")
                ds.close()
                return True
            except Exception as e:
                print(f"   ✗ Could not read file with xarray: {e}")
                return False
        else:
            print(f"   ✗ File was not created")
            return False

    except Exception as e:
        print(f"   ✗ Connection test failed: {e}")
        print("\n   Common issues:")
        print("   - Credentials file (~/.cdsapirc) not found or invalid")
        print("   - Invalid API key (check UID and key are correct)")
        print("   - Network connection problem")
        print("   - Copernicus CDS server is down (check https://cds.climate.copernicus.eu)")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Copernicus CDS API Connection Test")
    print("=" * 60)

    results = {
        'cdsapi_installed': check_cdsapi_installed(),
        'credentials_exists': check_credentials_file(),
    }

    # Only test connection if prerequisites are met
    if results['cdsapi_installed'] and results['credentials_exists']:
        results['connection'] = test_cds_connection()
    else:
        print("\n⚠ Skipping connection test (prerequisites not met)")
        results['connection'] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    status_symbol = lambda success: "✓" if success else "✗"

    print(f"{status_symbol(results['cdsapi_installed'])} cdsapi installed: {results['cdsapi_installed']}")
    print(f"{status_symbol(results['credentials_exists'])} Credentials file exists: {results['credentials_exists']}")

    if results['connection'] is not None:
        print(f"{status_symbol(results['connection'])} CDS connection works: {results['connection']}")

    # Final verdict
    all_passed = all(v for k, v in results.items() if v is not None)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to download data!")
        print("\nNext steps:")
        print("  1. Review the downloaded test_era5_sample.nc file")
        print("  2. Implement your data download pipeline in src/climate_risk/download.py")
        print("  3. Run: python -m climate_risk.download --period historical")
    else:
        print("✗ TESTS FAILED - Fix issues above before downloading data")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
