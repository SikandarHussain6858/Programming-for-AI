import pandas as pd
import numpy as np
import re

print("=" * 80)
print("TASK 3: TEXT PROCESSING WITH REGEX")
print("=" * 80)

# Load the housing dataset
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(housing_url)

print(f"\n[SUCCESS] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Create synthetic text columns for regex demonstration
print("\n" + "=" * 80)
print("CREATING SYNTHETIC TEXT DATA")
print("=" * 80)

np.random.seed(42)

# 1. Phone numbers with area codes
area_codes = ['415', '510', '650', '408', '925']
df['contact_phone'] = df.apply(
    lambda x: f"({np.random.choice(area_codes)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}", 
    axis=1
)

# 2. Email addresses with different domains
domains = ['gmail.com', 'yahoo.com', 'housing.gov', 'realestate.com']
df['agent_email'] = df.apply(
    lambda x: f"agent{np.random.randint(1000,9999)}@{np.random.choice(domains)}", 
    axis=1
)

# 3. Property codes (state-city-number format)
df['property_code'] = df.apply(
    lambda x: f"CA-{np.random.choice(['SF','LA','SD','OC','SJ'])}-{np.random.randint(10000,99999)}", 
    axis=1
)

print("\n[SUCCESS] Created 3 synthetic text columns:")
print("   1. contact_phone - Phone numbers with area codes")
print("   2. agent_email - Email addresses with various domains")
print("   3. property_code - Property codes in CA-CITY-NUMBER format")

print("\nSample data:")
print("-" * 80)
print(df[['contact_phone', 'agent_email', 'property_code']].head(10))

# TASK 3.1: EXTRACT SPECIFIC PATTERNS
print("\n" + "=" * 80)
print("1. REGEX PATTERN EXTRACTION")
print("=" * 80)

# Extract area codes from phone numbers
df['area_code'] = df['contact_phone'].str.extract(r'\((\d{3})\)')
print("\nExtracted area codes from phone numbers using pattern: r'\\((\\d{3})\\)'")
print("-" * 80)
print(df[['contact_phone', 'area_code']].head(10))

# Extract domain from emails
df['email_domain'] = df['agent_email'].str.extract(r'@(.+)')
print("\nExtracted email domains using pattern: r'@(.+)'")
print("-" * 80)
print(df[['agent_email', 'email_domain']].head(10))

# Extract city code from property codes
df['city_code'] = df['property_code'].str.extract(r'CA-([A-Z]{2})-')
print("\nExtracted city codes from property codes using pattern: r'CA-([A-Z]{2})-'")
print("-" * 80)
print(df[['property_code', 'city_code']].head(10))

# TASK 3.2: REPLACE/REMOVE UNWANTED CHARACTERS
print("\n" + "=" * 80)
print("2. REGEX REPLACEMENT/REMOVAL")
print("=" * 80)

# Remove parentheses, dashes, and spaces from phone numbers
df['phone_cleaned'] = df['contact_phone'].str.replace(r'[()\s-]', '', regex=True)
print("\nRemoved formatting characters from phone numbers using: r'[()\\s-]'")
print("-" * 80)
print(df[['contact_phone', 'phone_cleaned']].head(10))

# Standardize email to lowercase
df['email_standardized'] = df['agent_email'].str.lower()
print("\nStandardized email addresses to lowercase")
print("-" * 80)
print(df[['agent_email', 'email_standardized']].head(10))

# Remove dashes from property codes
df['property_simple'] = df['property_code'].str.replace(r'-', '', regex=True)
print("\nRemoved dashes from property codes using: r'-'")
print("-" * 80)
print(df[['property_code', 'property_simple']].head(10))

# TASK 3.3: VALIDATE FORMAT CONSISTENCY
print("\n" + "=" * 80)
print("3. REGEX FORMAT VALIDATION")
print("=" * 80)

# Validate phone number format: (XXX) XXX-XXXX
df['phone_valid'] = df['contact_phone'].str.match(r'^\(\d{3}\) \d{3}-\d{4}$')
valid_phones = df['phone_valid'].sum()
print(f"\nValidating phone format: (XXX) XXX-XXXX")
print(f"Pattern: r'^\\(\\d{{3}}\\) \\d{{3}}-\\d{{4}}$'")
print("-" * 80)
print(f"Valid phone numbers: {valid_phones:,} out of {len(df):,} ({valid_phones/len(df)*100:.1f}%)")
print(df[['contact_phone', 'phone_valid']].head(10))

# Validate email format: basic email pattern
df['email_valid'] = df['agent_email'].str.match(r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]+$')
valid_emails = df['email_valid'].sum()
print(f"\nValidating email format: text@domain.extension")
print(f"Pattern: r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-z]+$'")
print("-" * 80)
print(f"Valid email addresses: {valid_emails:,} out of {len(df):,} ({valid_emails/len(df)*100:.1f}%)")
print(df[['agent_email', 'email_valid']].head(10))

# Validate property code format: CA-XX-XXXXX
df['property_valid'] = df['property_code'].str.match(r'^CA-[A-Z]{2}-\d{5}$')
valid_properties = df['property_valid'].sum()
print(f"\nValidating property code format: CA-XX-XXXXX")
print(f"Pattern: r'^CA-[A-Z]{{2}}-\\d{{5}}$'")
print("-" * 80)
print(f"Valid property codes: {valid_properties:,} out of {len(df):,} ({valid_properties/len(df)*100:.1f}%)")
print(df[['property_code', 'property_valid']].head(10))

# SUMMARY
print("\n" + "=" * 80)
print("TASK 3 SUMMARY")
print("=" * 80)

print("\nRegex Operations Completed:")
print("\n1. EXTRACTION:")
print("   - Extracted area codes from phone numbers")
print("   - Extracted domains from email addresses")
print("   - Extracted city codes from property codes")

print("\n2. REPLACEMENT/REMOVAL:")
print("   - Cleaned phone numbers (removed formatting)")
print("   - Standardized emails (converted to lowercase)")
print("   - Simplified property codes (removed dashes)")

print("\n3. VALIDATION:")
print(f"   - Phone format validation: {valid_phones:,}/{len(df):,} valid")
print(f"   - Email format validation: {valid_emails:,}/{len(df):,} valid")
print(f"   - Property code validation: {valid_properties:,}/{len(df):,} valid")

print(f"\nNew columns created: 9")
print("   Extracted: area_code, email_domain, city_code")
print("   Cleaned: phone_cleaned, email_standardized, property_simple")
print("   Validated: phone_valid, email_valid, property_valid")

print(f"\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 80)
