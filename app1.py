import os
from random import randint
from datetime import datetime, timedelta

# Specify the remote branch
remote_branch = "main"  # Change this to your desired branch name

# Start from January 1, 2024
start_date = datetime(2024, 1, 1)

# Loop through each day of the year 2024 (365 days)
for i in range(365):  # Loop through 365 days of 2024
    commit_date = start_date + timedelta(days=i)  # Get the actual date for each day in 2024

    for j in range(randint(1, 10)):  # Random number of commits per day
        d = commit_date.strftime("%Y-%m-%d %H:%M:%S")  # Format the date in "YYYY-MM-DD HH:MM:SS"
        
        with open('file.txt', 'a') as file:
            file.write(f"Commit for {d}\n")  # Write the commit date to the file

        os.system('git add .')  # Stage all changes
        os.system(f'git commit --date="{d}" -m "commit for {d}"')  # Commit with the specific date

# Push to the specified branch
os.system(f'git push -u origin {remote_branch}')
