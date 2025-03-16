import os
from random import randint

# Specify the remote branch
remote_branch = "development"  # Change this to your desired branch name

for i in range(1, 365):  # Loop through 365 days
    for j in range(0, randint(1, 10)):  # Random number of commits per day
        d = str(i) + ' days ago'
        with open('file.txt', 'a') as file:
            file.write(d + '\n')  # Write the days-ago string to file with a newline
        
        os.system('git add .')  # Stage all changes
        os.system('git commit --date="' + d + '" -m "commit"')  # Commit with backdate

# Push to the specified branch
os.system('git push -u origin main')