import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from time import sleep

def web_spider(max_page):
    driver = webdriver.Chrome('/Users/Chris/Downloads/chromedriver')
    page = 1
    url = 'https://www.walmart.com/reviews/product/54742302'
    driver.get(url)

    titles_data = []
    dates_data = []
    yn_data = []
    reviews_data = []
    while page < max_page:

        if page in [1,2,3,4]:
            driver.find_element_by_xpath('//*[@id="CustomerReviews-container"]/div[1]/div[3]/div/div[1]/ul/li['+str(page)+']/button').click()
        elif page ==35:
            driver.find_element_by_xpath('//*[@id="CustomerReviews-container"]/div[1]/div[3]/div/div[1]/ul/li[6]/button').click()
        elif page ==36:
            driver.find_element_by_xpath('//*[@id="CustomerReviews-container"]/div[1]/div[3]/div/div[1]/ul/li[7]/button').click()
        else:
            driver.find_element_by_xpath('//*[@id="CustomerReviews-container"]/div[1]/div[3]/div/div[1]/ul/li[5]/button').click()

        sleep(3)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        titles = soup.findAll('div', attrs={'class': 'review-title'})
        dates = soup.findAll('div', attrs={'class':'review-submissionTime text-right'})
        numYesNo = soup.findAll('span', attrs={'class':'font-normal'})
        reviewsObj = soup.find_all('div', attrs={'class':'zeus-collapsable clearfix'}, string=True)

        tempTitle = [elem.text for elem in titles]
        tempDate = [elem.text for elem in dates]
        tempLike = [elem.text for elem in numYesNo]
        tempReview = [elem.text for elem in reviewsObj]

        titles_data.append(tempTitle)
        dates_data.append(tempDate)
        yn_data.append(tempLike)
        reviews_data.append(tempReview)

        # move to next page
        page += 1
        sleep(7)


    return [titles_data,dates_data,yn_data,reviews_data]

def main():
    walmart_scrap = web_spider(37) # scrapping website

    # unlist of result
    titles = sum([*walmart_scrap[0]], [])
    dates = sum([*walmart_scrap[1]], [])
    likes = sum([*walmart_scrap[2]], [])
    reviews = sum([*walmart_scrap[3]], [])

    # pre-processing of Yes and No number of counts
    like_data = []
    for i in range(len(likes)): # 1-step
        if i % 4 == 2:
            pass
        else:
            like_data.append(likes[i])

    final_like_data = []
    for i in range(len(like_data)): # 2-step
        if i % 3 == 2:
            pass
        else:
            final_like_data.append(like_data[i])

    yes = []
    no = []
    for i in range(len(final_like_data)): # final-step
        if i % 2 == 0:
            yes.append(final_like_data[i])
        else:
            no.append(final_like_data[i])

"""
//*[@id="Collapsable-1523332119652"]/div
//*[@id="Collapsable-1523332119657"]/div
//*[@id="Collapsable-1523332119663"]/div
//*[@id="Collapsable-1523332119666"]/div
//*[@id="Collapsable-1523332119669"]/div
//*[@id="Collapsable-1523332119671"]/div
//*[@id="Collapsable-1523332119675"]/div
//*[@id="CustomerReviews-container"]/div[1]/div[3]/div/div[1]/ul/li[1]/button
"""