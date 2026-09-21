#!/bin/bash

# ১. সব ফাইল যুক্ত করা
git add .

# ২. কমিট মেসেজ ইনপুট নেওয়া (অথবা ডিফল্ট মেসেজ দেওয়া)
echo "Enter commit message (or press Enter for default):"
read commit_msg

if [ -z "$commit_msg" ]; then
  commit_msg="Auto update from Termux"
fi

git commit -m "$commit_msg"

# ৩. গিটহাবে পুশ করা
git push origin main
