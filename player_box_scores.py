import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sqlite3
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests   
import shutil      
import datetime
from scipy.stats import norm
import os
import winsound

home_folder = 'C:\\Users\\Travis\\OneDrive\\Data Science\\Personal_Projects\\Sports\\NBA_Prediction_V3_1'
os.chdir(home_folder)

######   Functions   ######
def replace_name_values2(filename):
        # replace values with dashes for compatibility
    filename = filename.replace('%','_')
    filename = filename.replace('=','_')
    filename = filename.replace('?','_')
    filename = filename.replace('&','_')
    filename = filename.replace('20Season_','')
    filename = filename.replace('_20Season','')
    filename = filename.replace('SeasonType_','')
    filename = filename.replace('sort_gdate_dir_-1_','')
    filename = filename.replace('SeasonYear_','')
    return filename

def grab_player_data2(url_list, file_folder):    
        
        # Scrape Season-Level player data from the url_list

        i = 0
        for u in url_list:
                
                driver.get(u)
                time.sleep(2)

                # if the page does not load, go to the next in the list
                try:
                        xpath = '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select/option[1]'
                        elem = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
                except:
                        print(f'{u} did not load. Moving to next url.')
                        continue

                # click "all pages"
                xpath_all = '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[3]/div/label/div/select/option[1]' 
                elem = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath_all)))
                
                driver.find_element(by=By.XPATH, value=xpath_all).click()
                src = driver.page_source
                parser = BeautifulSoup(src, "lxml")
                table = parser.find("table", attrs = {"class":"Crom_table__p1iZz"})
                headers = table.findAll('th')
                headerlist = [h.text.strip() for h in headers[0:]] 
                row_names = table.findAll('a')                             # find rows
                row_list = [b.text.strip() for b in row_names[0:]] 
                rows = table.findAll('tr')[0:]
                player_stats = [[td.getText().strip() for td in rows[i].findAll('td')[0:]] for i in range(len(rows))]
                tot_cols = len(player_stats[1])                           #set the length to ignore hidden columns
                headerlist = headerlist[:tot_cols]   
                stats = pd.DataFrame(player_stats, columns = headerlist)

                # assign filename
                filename = file_folder + str(u[34:]).replace('/', '_') + '.csv'
                filename = replace_name_values2(filename)
                pd.DataFrame.to_csv(stats, filename)
                i += 1
                lu = len(url_list)
                # close driver
                print(f'{filename} Completed Successfully! {i} / {lu} Complete!')

        winsound.Beep(523, 500)

def append_the_data(folder, data_prefix, filename_selector):
    # Appending data together via folder and/or file name

    path = folder
    p = os.listdir(path)
    pf = pd.DataFrame(p)


    # filter for files that contain the filename_selector
    pf_reg = pf.loc[pf[0].astype(str).str.contains(filename_selector)] 

    appended_data = []
    for file in pf_reg[0]:
        data = pd.read_csv(folder + '/' + file)
        # if "Season" a column, drop it
        if 'Season' in data.columns:
            data = data.drop(columns = ['Season'])
        
        data['season'] = file[(file.find('20')):(file.find('20'))+4]
        data['season_type'] = np.where('Regular' in file, 'Regular', 'Playoffs')
        # add prefix to columns
        data = data.add_prefix(data_prefix)
        data.columns = data.columns.str.lower()
        appended_data.append(data)
    
    appended_data = pd.concat(appended_data)
    return appended_data

# Date
today = datetime.date.today()
month = today.month


# read current boxscores, get the last date
boxscores = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')
boxscores['date'] = pd.to_datetime(boxscores['trad_game date'])
boxscores = boxscores.sort_values(by = 'date', ascending = False)
boxscores = boxscores.reset_index(drop = True)
last_date = boxscores['date'][0]
# change to date from datetime
last_date = last_date.date()
print(f' last game date in df: {last_date}')


# get months since oct 2022
today = datetime.date.today()
months_since_oct_2022 = (today.year - 2022) * 12 + today.month - 9
this_season_months = np.arange(1, months_since_oct_2022+1, 1)
current_month = max(this_season_months)


def get_all_urls():
    # List all months to scrape
    urls1 = []

    for m in months_2022:
        for t in types:
            url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2022-23&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
            urls1.append(url)

    for m in months_2021:
        for t in types: 
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2021-22&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2020:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2020-21&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2019: 
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2019-20&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2018:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2018-19&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2017:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2017-18&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2016:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2016-17&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2015:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2015-16&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2014:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2014-15&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2013:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2013-14&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    for m in months_2012:
        for t in types:
                url = 'https://www.nba.com/stats/players/' + str(t) +'/?Season=2012-13&sort=gdate&dir=-1&Month='+str(m) +'&SeasonType=Regular%20Season'
                urls1.append(url)

    return urls1

# if the last_date is before yesterday, then scrape the new data
yesterday = today - datetime.timedelta(days=1)

if last_date < yesterday:
    driver = webdriver.Chrome()
    # minimize the window
    driver.minimize_window()
    # update this month's data
    trad_url_update = ('https://www.nba.com/stats/players/boxscores-traditional/?Season=2022-23&sort=gdate&dir=-1&Month='+str(current_month) +'&SeasonType=Regular%20Season')
    adv_url_update = ('https://www.nba.com/stats/players/boxscores-advanced/?Season=2022-23&sort=gdate&dir=-1&Month='+str(current_month) +'&SeasonType=Regular%20Season')
    urls2 = [trad_url_update, adv_url_update]
    # scrape trad and adv updates
    grab_player_data2(urls2, 'data/player/box_scores/')
    all_boxes_trad = append_the_data('data/player/box_scores/', 'trad_', 'traditional')
    all_boxes_adv = append_the_data('data/player/box_scores/', 'adv_', 'advanced')

    all_boxes = pd.merge(all_boxes_trad,all_boxes_adv,
                            left_on=['trad_player', 'trad_team', 'trad_season', 'trad_season_type', 'trad_match up', 'trad_game date'],
                            right_on=['adv_player', 'adv_team', 'adv_season', 'adv_season_type', 'adv_match up', 'adv_game date'],
                            how='left')
    all_boxes = all_boxes.dropna(subset=['trad_player'])

    today = datetime.datetime.today().strftime('%Y-%m-%d')

    all_boxes.to_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

    print('Data is Now Updated')

else:
    print('Data was Already Updated')
     