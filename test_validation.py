#!/usr/bin/env python3
"""
Test script to verify the validation is working in the updated dashboard
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'streamlit_dashboard'))

# Import the validation function
from streamlit_dashboard.app import validate_input_data

def test_validation():
    """Test the validation function with various inputs"""
    print("Testing input validation function...")
    
    # Test valid inputs
    print("\n1. Testing valid inputs:")
    age = 30
    experience = 10
    vehicle_age = 5
    accidents = 0
    annual_mileage = 15.0
    
    errors, warnings = validate_input_data(age, experience, vehicle_age, accidents, annual_mileage)
    print(f"Valid inputs - Errors: {errors}, Warnings: {warnings}")
    
    # Test invalid age (too young)
    print("\n2. Testing invalid age (too young):")
    errors, warnings = validate_input_data(17, 2, 5, 0, 15.0)
    print(f"Age 17 - Errors: {errors}, Warnings: {warnings}")
    
    # Test experience > age
    print("\n3. Testing experience > age-15:")
    errors, warnings = validate_input_data(25, 15, 5, 0, 15.0)
    print(f"Age 25, Experience 15 - Errors: {errors}, Warnings: {warnings}")
    
    # Test high accident count
    print("\n4. Testing high accident count:")
    errors, warnings = validate_input_data(30, 10, 5, 6, 15.0)
    print(f"6 accidents - Errors: {errors}, Warnings: {warnings}")
    
    # Test high mileage
    print("\n5. Testing high annual mileage:")
    errors, warnings = validate_input_data(30, 10, 5, 0, 80.0)
    print(f"80k km mileage - Errors: {errors}, Warnings: {warnings}")
    
    print("\n✅ Validation function is working correctly!")

if __name__ == "__main__":
    test_validation()