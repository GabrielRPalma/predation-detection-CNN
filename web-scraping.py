# Importing section
from selenium import webdriver
import os
# End importing section

# Selecting search items
searchterm = 'tiger+predation'# Add your items as this exemple
# End Selecting search items

# Looking for these items in google with selenium
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
browser = webdriver.Chrome("/home/gabriel/Desktop/Writting/My_papers/IMVIP/images/chromedriver")#insert path to chromedriver inside parentheses
browser.get(url)
# End Looking for these items in google with selenium

# Choosing the image extensions
img_count = 0
extensions = { "jpg", "jpeg", "png", "gif" }
# End # Choosing the image extensions

# Add a new directory to story the images
if not os.path.exists(searchterm):
    os.mkdir(searchterm)
# End Add a new directory to story the images

# Section scrolling in google
for _ in range(500):
    browser.execute_script("window.scrollBy(0,10000)")
# End Section scrolling in google    

# Collecting the image path
html = browser.page_source.split('["')
images = []
for i in html:
    if i.startswith('http') and i.split('"')[0].split('.')[-1] in extensions:
        imges.append(i.split('"')[0])  
# End Collecting the image path

# Closing the browser
browser.close()
# End Closing the browser

# Storing the collected image paths
f = open('path_store.txt', 'w')
for i in range(len(images)):

    f.write(imges[i] + '\n')

f.close()
# End Storing the collected image paths

# After producing the path_store.txt you should use the following command in linux to store the images 
# wget -i path_store.txt -P your_path_to/searchterm # You can choose other directory to store the images instead of searchterm
