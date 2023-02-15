import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC   
import datetime
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from selenium.common.exceptions import WebDriverException
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import sklearn as sk
from sklearn.metrics import r2_score

import plotly.figure_factory as ff

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)

today = pst.strftime('%Y-%m-%d')
st.write('the current date is: ' + today)

# Load Data
player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')
# add capitalized player name
player_numbers['Player'] = player_numbers['player_name'].str.title()

# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# Load tracking and other data
catch_shoot = pd.read_csv('data/player/nba_com_playerdata/tracking/catch_shoot_' + today + '_.csv')
defensive_impact = pd.read_csv('data/player/nba_com_playerdata/tracking/defensive_impact_' + today + '_.csv')
drives = pd.read_csv('data/player/nba_com_playerdata/tracking/drives_' + today + '_.csv')
elbow_touches = pd.read_csv('data/player/nba_com_playerdata/tracking/elbow_touches_' + today + '_.csv')
paint_touches = pd.read_csv('data/player/nba_com_playerdata/tracking/paint_touches_' + today + '_.csv')
passing = pd.read_csv('data/player/nba_com_playerdata/tracking/passing_' + today + '_.csv')
pull_up_shooting = pd.read_csv('data/player/nba_com_playerdata/tracking/pull_up_shots_' + today + '_.csv')
rebounding = pd.read_csv('data/player/nba_com_playerdata/tracking/rebounds_' + today + '_.csv')
speed_distance = pd.read_csv('data/player/nba_com_playerdata/tracking/speed_distance_' + today + '_.csv')
touches = pd.read_csv('data/player/nba_com_playerdata/tracking/touches_' + today + '_.csv')
shooting_efficiency = pd.read_csv('data/player/nba_com_playerdata/tracking/shooting_efficiency_' + today + '_.csv')

# Load Shooting Data
catch_shoot_shooting = pd.read_csv('data/player/nba_com_playerdata/shooting/catch_and_shoot_' + today + '_.csv')
opp_shooting_5ft = pd.read_csv('data/player/nba_com_playerdata/shooting/opp_shooting_5ft_' + today + '_.csv')
opp_shooting_by_zone = pd.read_csv('data/player/nba_com_playerdata/shooting/opp_shooting_by_zone_' + today + '_.csv')
pullups = pd.read_csv('data/player/nba_com_playerdata/shooting/pullups_' + today + '_.csv')
shooting_splits_5ft = pd.read_csv('data/player/nba_com_playerdata/shooting/shooting_splits_5ft_' + today + '_.csv')
shooting_splits_by_zone = pd.read_csv('data/player/nba_com_playerdata/shooting/shooting_splits_by_zone_' + today + '_.csv')
shot_dash_general = pd.read_csv('data/player/nba_com_playerdata/shooting/shot_dash_general_' + today + '_.csv')



# check last date
gbg_df['Date'] = pd.to_datetime(gbg_df['trad_game date'])
# change datetime to date
gbg_df['Date'] = gbg_df['Date'].dt.date
# sort by date
gbg_df = gbg_df.sort_values(by = 'Date', ascending = False)
# get last date
last_date = gbg_df['Date'].iloc[0]
st.sidebar.write('Last Date of Box Score Data: ' + str(last_date))

st.title('NBA Player Analytics Tool')

st.write('This application compares a player\'s performance to the league average and to other players at their position. The data is pulled from a variety of sources, and is updated daily. ')



st.subheader('Player Size Data')


# select team
teams = gbg_df['trad_team'].unique()
# sort teams
teams = np.sort(teams)
team = st.sidebar.selectbox('Select Team', teams, index = 2)

# select player
gbg_22 = gbg_df[gbg_df['adv_season'] == 2022]
players_22 = gbg_22[gbg_22['trad_team'] == team]['trad_player'].unique()
# sort players
players_22 = np.sort(players_22)
player = st.sidebar.selectbox('Select Player', players_22, index = 12)

player_nba_id = player_numbers[player_numbers['Player'] == player]['nba_id'].iloc[0]

st.sidebar.write('Player NBA_id: ' + str(player_nba_id))

player_photo = 'data/player/photos/photos/' + str(player_nba_id) + '.png'
# add player photo to sidebar
st.sidebar.image(player_photo, width = 200)


# select position
position_options = ['PG', 'SG', 'SF', 'PF', 'C']
position = st.sidebar.selectbox('Select Position to evaluate the player at', position_options)

# load player data
player_gbg = gbg_df[gbg_df['trad_player'] == player]
unnamed_cols = [col for col in player_gbg.columns if 'Unnamed' in col or 'unnamed' in col]
player_gbg = player_gbg.drop(columns = unnamed_cols)


# load player size data
player_size = df_sizes[df_sizes['player'] == player]
keepcols = ['player', 'height_final', 'wingspan_final', 'primary_position_bbref']


# calculate median height and wingspan for position with df_sizes in last 5 years
df_sizes = df_sizes[df_sizes['position_season'] >= 2017]


median_height = df_sizes[df_sizes['primary_position_bbref'] == position]['height_final'].median()
median_wingspan = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan_final'].median()

# identify player height and wingspan
player_size = player_size[keepcols]
player_height = player_size['height_final'].iloc[0]
player_wingspan = player_size['wingspan_final'].iloc[0]

# make a df of just players and their primary position
primary_positions = df_sizes[['player', 'primary_position_bbref', 'position_season']]

# calculate player height percentile using df_sizes and primary position
# drop any heights that are 0
df_sizes = df_sizes[df_sizes['height_final'] > 0]

# get all heights for position
all_heights = df_sizes[df_sizes['primary_position_bbref'] == position]['height_final']
all_wingspans = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan_final']

# calculate wingspan over height in df_sizes
df_sizes['wingspan__height_ratio'] = df_sizes['wingspan_final'] / df_sizes['height_final']
all_wingspan_height_ratios = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan__height_ratio']

# get percentile of player height
player_height_percentile = norm.cdf(player_height, all_heights.mean(), all_heights.std()) * 100
player_wingspan_percentile = norm.cdf(player_wingspan, all_wingspans.mean(), all_wingspans.std()) * 100
player_wingspan_height_ratio_percentile = norm.cdf(player_wingspan / player_height, all_wingspan_height_ratios.mean(), all_wingspan_height_ratios.std()) * 100



# 3 columns
col1, col2, col3 = st.columns(3)
# display player size data


def number_post(num):
    # get the last digit of the number
    last_digit = str(num)[-1]
    # if the last digit is 1, return st
    if last_digit == '1':
        return 'st'
    # if the last digit is 2, return nd
    elif last_digit == '2':
        return 'nd'
    # if the last digit is 3, return rd
    elif last_digit == '3':
        return 'rd'
    # if the last digit is 4-9, return th
    else:
        return 'th'


col1.metric('Player Height', str(player_height) + ' inches',  delta_color='off')
col1.write(player + ' is in the **' + str(int(player_height_percentile)) + number_post(int(player_height_percentile))+'** percentile for height at the ' + position + ' position.')

def color_def():
    if player_height_percentile < 50:
        return 'red'
    else:
        return 'green'

# plot small bar chart for height percentile with plotly. Color is red if below median, green if above median
fig = go.Figure(go.Bar(x = [player_height_percentile], y = ['Height Percentile'], orientation = 'h', marker_color = color_def()))
fig.update_layout(title = position + ' Height Percentile for ' + player, width = 400, height = 200)
# show the whole bar chart
fig.update_xaxes(range = [0, 100])
# get rid of y axis
fig.update_yaxes(showticklabels = False)
col1.plotly_chart(fig, use_container_width = True)


# plot height distribution with player height indicated with seaborn

fig, ax = plt.subplots(figsize = (10, 5))
sns.distplot(all_heights, ax = ax, bins = 20, color = 'green')
ax.axvline(player_height, color = 'red', linestyle = '--')
ax.set_title('Height Distribution for ' + position + 's')
ax.set_xlabel('Height (inches)')
ax.set_ylabel('Density')
col1.pyplot(fig, use_container_width = True)

col2.metric('Player Wingspan', str(player_wingspan) + ' inches')
col2.write(player + ' is in the **' + str(int(player_wingspan_percentile)) + number_post(int(player_wingspan_percentile))+'** percentile for wingspan at the ' + position + ' position.')

def color_def():
    if player_wingspan_percentile < 50:
        return 'red'
    else:
        return 'green'

# plot small bar chart for wingspan percentile with plotly. Color is red if below median, green if above median
fig = go.Figure(go.Bar(x = [player_wingspan_percentile], y = ['Wingspan Percentile'], orientation = 'h', marker_color = color_def()))
fig.update_layout(title = position + ' Wingspan Percentile for ' + player, width = 400, height = 200)
# show the whole bar chart
fig.update_xaxes(range = [0, 100])
# get rid of y axis
fig.update_yaxes(showticklabels = False)
col2.plotly_chart(fig, use_container_width = True)



# plot wingspan distribution with player wingspan indicated
fig, ax = plt.subplots(figsize = (10, 5))
sns.distplot(all_wingspans, ax = ax, bins = 20, color = 'green')
ax.axvline(player_wingspan, color = 'red', linestyle = '--')
ax.set_title('Wingspan Distribution for ' + position + 's')
ax.set_xlabel('Wingspan (inches)')
ax.set_ylabel('Density')
col2.pyplot(fig, use_container_width = True)


col3.metric('Player Wingspan / Height Ratio', str(round(player_wingspan / player_height,2)))
col3.write(player + ' is in the **' + str(int(player_wingspan_height_ratio_percentile)) + number_post(int(player_wingspan_height_ratio_percentile))+'** percentile for wingspan / height ratio at the ' + position + ' position.')

def color_def():
    if player_wingspan_height_ratio_percentile < 50:
        return 'red'
    else:
        return 'green'

# plot small bar chart for wingspan / height ratio percentile with plotly. Color is red if below median, green if above median
fig = go.Figure(go.Bar(x = [player_wingspan_height_ratio_percentile], y = ['Wingspan / Height Ratio Percentile'], orientation = 'h', marker_color = color_def()))
fig.update_layout(title = position + ' Wingspan / Height Ratio Percentile for ' + player, width = 400, height = 200)
# show the whole bar chart
fig.update_xaxes(range = [0, 100])
# get rid of y axis
fig.update_yaxes(showticklabels = False)
col3.plotly_chart(fig, use_container_width = True)



# plot wingspan / height ratio distribution with player wingspan / height ratio indicated
fig, ax = plt.subplots(figsize = (10, 5))
sns.distplot(all_wingspan_height_ratios, ax = ax, bins = 20, color = 'green')
ax.axvline(player_wingspan / player_height, color = 'red', linestyle = '--')
ax.set_title('Wingspan / Height Ratio Distribution for ' + position + 's')
ax.set_xlabel('Wingspan / Height Ratio')
ax.set_ylabel('Density')
col3.pyplot(fig, use_container_width = True)

# div
st.markdown('---')


st.subheader('Player Game Data')

player_gbg_22 = player_gbg[player_gbg['adv_season'] == 2022]
# get season averages
player_ppg = player_gbg_22['trad_pts'].mean()
player_rpg = player_gbg_22['trad_reb'].mean()
player_apg = player_gbg_22['trad_ast'].mean()
player_tovpg = player_gbg_22['trad_tov'].mean()
player_3p_pct = player_gbg_22['trad_3pm'].sum() / player_gbg_22['trad_3pa'].sum() *100
player_fg_pct = player_gbg_22['trad_fgm'].sum() / player_gbg_22['trad_fga'].sum() *100

# display season averages as metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric('Points Per Game', str(round(player_ppg, 2)))
col2.metric('Rebounds Per Game', str(round(player_rpg, 2)))
col3.metric('Assists Per Game', str(round(player_apg, 2)))
col4.metric('Turnovers Per Game', str(round(player_tovpg, 2)))
col5.metric('3 Point %', str(round(player_3p_pct, 2)))
col6.metric('FG %', str(round(player_fg_pct, 2)))

# add home or away column
player_gbg['Home'] = np.where(player_gbg['trad_match up'].str.contains('vs'), 1, 0)
# add points per minute column
player_gbg['ppm'] = player_gbg['trad_pts'] / player_gbg['trad_min']
# fix columns names, drop 'trad'
player_gbg.columns = player_gbg.columns.str.replace('trad_', '')
# get numeric columns
numeric_cols= player_gbg.select_dtypes(include = ['float64', 'int64']).columns
to_drop = ['adv_match up', 'adv_game date', 'adv_w/l', 'adv_player', 'adv_team', 'adv_season_type']
player_gbg = player_gbg.drop(columns = to_drop)
# set index to player
player_gbg = player_gbg.set_index('player')

st.dataframe(player_gbg.style.format('{:.1f}', subset = numeric_cols))

# calculate averages for player
player_gbg_avg = player_gbg.groupby('player').mean().reset_index()
 
 # show efg% and ts%, 3p%, fg%, adv_pace
player_gbg_avg = player_gbg_avg[['player', 'adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']]
ncols = ['adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']
# get rid of index
player_gbg_avg = player_gbg_avg.set_index('player')
# rename columns
player_gbg_avg.columns = ['Offensive Rating', 'Defensive Rating', 'Effective FG%', 'True Shooting%', 'Pace', 'Points Per Minute']
st.write('**Advanced Averages**')
st.table(player_gbg_avg.style.format('{:.2f}', subset = player_gbg_avg.columns))


# make sure position_season is 2022
primary_positions = primary_positions[primary_positions['position_season'] == 2022]
# drop duplicates
primary_positions = primary_positions.drop_duplicates(subset = 'player')

# add position column to gbg_22
gbg_22['position'] = gbg_22['trad_player'].map(primary_positions.set_index('player')['primary_position_bbref'])

# compare player to position

# get averages for player position
position_gbg = gbg_22[gbg_22['position'] == position]
position_gbg_mean = position_gbg.groupby('position').mean().reset_index()


# three columns
st.subheader('Player Game Data Plots, Home vs Away')

col1, col2, col3 = st.columns(3)
# plot distributions of adv_ts%, 3p%, ppm with plotly
fig = px.histogram(player_gbg, x = 'adv_ts%', color = 'Home', marginal='box', nbins = 20, opacity = 0.3)
fig.update_layout(title = 'TS% Distribution for ' + player)
col1.plotly_chart(fig)

fig = px.histogram(player_gbg, x = '3p%', color = 'Home', marginal = 'box', nbins = 20, opacity = 0.3)
fig.update_layout(title = '3P% Distribution for ' + player)
col2.plotly_chart(fig)

fig = px.histogram(player_gbg, x = 'ppm', color = 'Home', marginal = 'box', nbins = 20, opacity = 0.3)
fig.update_layout(title = 'Points Per Minute Distribution for ' + player)
col3.plotly_chart(fig)

c1, c2 = st.columns(2)

# add a plotly ddistplot for player pts at home and away
home_pts = player_gbg[player_gbg['Home'] == 1]['pts']
away_pts = player_gbg[player_gbg['Home'] == 0]['pts']
fig = ff.create_distplot([home_pts, away_pts], ['Home', 'Away'], bin_size = 2)
fig.update_layout(title = 'Points Distribution for ' + player + ' at Home and Away')
c1.plotly_chart(fig, use_container_width=True)

# add cdf plot for player pts at home and away
fig = px.line(x = np.sort(home_pts), y = np.arange(1, len(home_pts) + 1) / len(home_pts), title = 'CDF of Points for ' + player + ' at Home')
fig.add_scatter(x = np.sort(away_pts), y = np.arange(1, len(away_pts) + 1) / len(away_pts), name = 'Away')
fig.update_layout(title = 'CDF of Points Scored for ' + player + ' at Home and Away')
c2.plotly_chart(fig, use_container_width=True)

st.markdown('---')

st.subheader('Shooting')
# 3 columns
col1, col2, col3 = st.columns(3)


col1.write('Shooting Efficiency')

player_shooting_efficiency_init = shooting_efficiency[shooting_efficiency['PLAYER'] == player]
# transpose the dataframe
player_shooting_efficiency = player_shooting_efficiency_init.T
# drop first 8 rows
player_shooting_efficiency = player_shooting_efficiency.iloc[8:]
# rename columns
player_shooting_efficiency.columns = ['Shooting Efficiency Metrics']

# add position column to shooting_efficiency
shooting_efficiency['position'] = shooting_efficiency['PLAYER'].map(primary_positions.set_index('player')['primary_position_bbref'])
# get averages for player position
position_shooting_efficiency = shooting_efficiency[shooting_efficiency['position'] == position]
position_shooting_efficiency_mean = position_shooting_efficiency.groupby('position').mean().reset_index()

# transpose, drop first 7 rows, add to player_shooting_efficiency
position_shooting_efficiency_mean = position_shooting_efficiency_mean.T
position_shooting_efficiency_mean = position_shooting_efficiency_mean.iloc[7:]
position_shooting_efficiency_mean.columns = ['Position Average']
player_shooting_efficiency2 = pd.concat([player_shooting_efficiency, position_shooting_efficiency_mean], axis = 1)

# if player is not in position_shooting_efficiency, add them
if player not in position_shooting_efficiency['PLAYER'].values:
    # add player_shooting_efficiency_init to position_shooting_efficiency
    position_shooting_efficiency = position_shooting_efficiency.append(player_shooting_efficiency_init)

# add percentile columns to position_shooting_efficiency
for col in position_shooting_efficiency.columns:
    if col != 'PLAYER' and col != 'position':
        position_shooting_efficiency[col + '_percentile'] = position_shooting_efficiency[col].rank(pct = True)

# get player percentile
player_percentile = position_shooting_efficiency[position_shooting_efficiency['PLAYER'] == player]

# add player percentile to player_shooting_efficiency2
player_percentile = player_percentile.T
player_percentile.columns = ['Player Percentile']
# drop first 29 rows
player_percentile = player_percentile.iloc[29:]
# rename all indexes, replacing '_percentile' with ''
player_percentile.index = [col.replace('_percentile', '') for col in player_percentile.index]

player_shooting_efficiency2 = pd.concat([player_shooting_efficiency2, player_percentile], axis = 1)

# function to color code Player Percentiles
def color_code_percentile(val):
    if val < 0.4:
        color = 'red'
    elif val < 0.45 and val > 0.4:
        color = 'orange'
    elif val < 0.55 and val > 0.45:
        color = 'white',
    elif val > 0.55 and val < 0.75:
        color = 'lightgreen'
    elif val > 0.75:
        color = 'green'
    else:
        color = 'white'
    # return highlight color (background)
    return 'background-color: %s' % color

col1.table(player_shooting_efficiency2.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))

#############################################################################################################


col2.write('Shooting by Zone')

# shooting_splits_by_zone
player_shooting_splits_by_zone_init = shooting_splits_by_zone[shooting_splits_by_zone['Player'] == player]

# assign final columns 
shooting_by_zone_final_cols = ['Restricted Area_FGA', 'Restricted Area_FG%', 'In The Paint (Non-RA)_FGA', 
                                'In The Paint (Non-RA)_FG%', 'Mid-Range_FGA', 'Mid-Range_FG%', 'Left Corner 3._FGA', 
                                'Left Corner 3._FG%', 'Right Corner 3._FGA', 'Right Corner 3._FG%', 'Above the Break 3._FGA', 
                                'Above the Break 3._FG%']

# assign final columns to player_shooting_splits_by_zone_init
player_shooting_splits_by_zone_init = player_shooting_splits_by_zone_init

# transpose the dataframe
player_shooting_splits_by_zone = player_shooting_splits_by_zone_init.T

# add position column to shooting_splits_by_zone
shooting_splits_by_zone['position'] = shooting_splits_by_zone['Player'].map(primary_positions.set_index('player')['primary_position_bbref'])

# get averages for player POSITION
position_shooting_splits_by_zone_init = shooting_splits_by_zone[shooting_splits_by_zone['position'] == position]

# change '-' values to 0
position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.replace('-', 0)

# assign final columns to position_shooting_splits_by_zone
position_shooting_splits_by_zone = position_shooting_splits_by_zone_init[shooting_by_zone_final_cols]

# make all values numeric
position_shooting_splits_by_zone = position_shooting_splits_by_zone.apply(pd.to_numeric, errors = 'coerce')
# make mean df
position_shooting_splits_by_zone_mean = position_shooting_splits_by_zone.mean().reset_index()


# add position average to player_shooting_splits_by_zone using left join
player_shooting_splits_by_zone = player_shooting_splits_by_zone.merge(position_shooting_splits_by_zone_mean, 
                                    how = 'left', left_index = True, right_on = 'index')

# reset index to index
player_shooting_splits_by_zone = player_shooting_splits_by_zone.set_index('index')

# Add Player Percentile
# if player is not in position_shooting_splits_by_zone, add them
if player not in position_shooting_splits_by_zone_init['Player'].values:
    # add player to position_shooting_splits_by_zone
    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.append(player_shooting_splits_by_zone_init)

# drop unnamed cols
unnamed = [col for col in position_shooting_splits_by_zone_init.columns if 'Unnamed' in col]
position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.drop(unnamed, axis = 1)
# drop team and position cols
position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.drop(['Team', 'position'], axis = 1)
# ADD percentile columns
# final cols plus player
final_cols_and_player = ['Player'] + shooting_by_zone_final_cols 

position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init[final_cols_and_player]

# make sure all values (other than index) are numeric
position_shooting_splits_by_zone_init[shooting_by_zone_final_cols] = position_shooting_splits_by_zone_init[shooting_by_zone_final_cols].apply(pd.to_numeric, errors = 'coerce')

for col in position_shooting_splits_by_zone_init.columns:
    # get percentile
    position_shooting_splits_by_zone_init[col + '_percentile'] = position_shooting_splits_by_zone_init[col].rank(pct = True)
# drop nan in player col
position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.dropna(subset = ['Player'])


# get player percentile
player_percentile = position_shooting_splits_by_zone_init[position_shooting_splits_by_zone_init['Player'] == player]
# drop all columns except percentile columns
player_percentile = player_percentile[[col for col in player_percentile.columns if 'percentile' in col]]

player_percentile_t = player_percentile.T

# drop any rows with nan
player_shooting_splits_by_zone = player_shooting_splits_by_zone.dropna()

# rename indexes, removing _percentile
player_percentile_t.index = [col.replace('_percentile', '') for col in player_percentile_t.index]


# add player percentile to player_shooting_splits_by_zone
player_shooting_splits_by_zone = player_shooting_splits_by_zone.merge(player_percentile_t, how = 'left', left_index = True, right_index = True)

# rename columns
player_shooting_splits_by_zone.columns = ['Player', 'Position Average', 'Player Percentile']
# make sure Player column is all numeric
player_shooting_splits_by_zone['Player'] = player_shooting_splits_by_zone['Player'].apply(pd.to_numeric, errors = 'coerce').round(2)

# make sure position average column is all numeric
player_shooting_splits_by_zone['Position Average'] = player_shooting_splits_by_zone['Position Average'].apply(pd.to_numeric, errors = 'coerce').round(2)

# show df
col2.table(player_shooting_splits_by_zone.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))


####################################################################################################################

col3.write('Shooting by Distance (5ft)')
# shooting_splits_5ft
player_shooting_splits_5ft_init = shooting_splits_5ft[shooting_splits_5ft['Player'] == player]
# drop fgm cols
fgm_cols = [col for col in player_shooting_splits_5ft_init.columns if 'FGM' in col]
player_shooting_splits_5ft = player_shooting_splits_5ft_init.drop(columns = fgm_cols)

# add transposed version of player_shooting_splits_5ft
player_shooting_splits_5ft_t = player_shooting_splits_5ft.T

# rename column to 'player metrics'
player_shooting_splits_5ft_t.columns = ['Player Metrics']

# add position column to shooting_splits_5ft
shooting_splits_5ft['position'] = shooting_splits_5ft['Player'].map(primary_positions.set_index('player')['primary_position_bbref'])

# add position_shooting_splits_5ft 
position_shooting_splits_5ft = shooting_splits_5ft[shooting_splits_5ft['position'] == position]
# replace - with 0
position_shooting_splits_5ft = position_shooting_splits_5ft.replace('-', 0)

# get columns besides player and position
cols = [col for col in position_shooting_splits_5ft.columns if col not in ['Player', 'position']]

#make sure values are numeric
position_shooting_splits_5ft[cols] = position_shooting_splits_5ft[cols].apply(pd.to_numeric, errors = 'coerce')
# get mean
position_shooting_splits_5ft_mean = position_shooting_splits_5ft.mean()


# rename column in position_shooting_splits_5ft_mean to Position Average
position_shooting_splits_5ft_mean = position_shooting_splits_5ft_mean.rename('Position Average')


# add position average to player_shooting_splits_5ft using left join
player_shooting_splits_5ft = pd.merge(player_shooting_splits_5ft_t, position_shooting_splits_5ft_mean, how = 'left', left_index = True, right_index = True)

# add player to position_shooting_splits_5ft IF they are not already in it
if player not in position_shooting_splits_5ft['Player'].values:
    position_shooting_splits_5ft = position_shooting_splits_5ft.append(player_shooting_splits_5ft_init)
    
# drop unnamed cols
unnamed_cols = [col for col in position_shooting_splits_5ft.columns if 'Unnamed' in col]
position_shooting_splits_5ft = position_shooting_splits_5ft.drop(columns = unnamed_cols)

# drop team and position cols
position_shooting_splits_5ft = position_shooting_splits_5ft.drop(['Team', 'position'], axis = 1)

# Make sure all values (other than Player column) are numeric
position_shooting_splits_5ft[position_shooting_splits_5ft.columns[1:]] = position_shooting_splits_5ft[position_shooting_splits_5ft.columns[1:]].apply(pd.to_numeric, errors = 'coerce')


# ADD percentile columns
for col in position_shooting_splits_5ft.columns:
    # get percentile
    position_shooting_splits_5ft[col + '_percentile'] = position_shooting_splits_5ft[col].rank(pct = True)


# get player percentile
player_percentile = position_shooting_splits_5ft[position_shooting_splits_5ft['Player'] == player]
# drop all columns except percentile columns
player_percentile = player_percentile[[col for col in player_percentile.columns if 'percentile' in col]]

# transpose player_percentile
player_percentile_t = player_percentile.T

# rename indexes, removing _percentile
player_percentile_t.index = [col.replace('_percentile', '') for col in player_percentile_t.index]

# rename column to Player Percentile
player_percentile_t.columns = ['Player Percentile']

# add player percentile to player_shooting_splits_5ft
player_shooting_splits_5ft = player_shooting_splits_5ft.merge(player_percentile_t, how = 'left', left_index = True, right_index = True)

# drop unnamed rows
unnamed_rows = [row for row in player_shooting_splits_5ft.index if 'Unnamed' in row]
player_shooting_splits_5ft = player_shooting_splits_5ft.drop(index = unnamed_rows)
# drop player and team rows
player_shooting_splits_5ft = player_shooting_splits_5ft.drop(['Player', 'Team'])


# make sure all values are numeric
player_shooting_splits_5ft[player_shooting_splits_5ft.columns] = player_shooting_splits_5ft[player_shooting_splits_5ft.columns].apply(pd.to_numeric, errors = 'coerce')


# show df
col3.table(player_shooting_splits_5ft.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))

####################################################################################################################

######### SHOT DASHBOARD #########

st.write('Shot Dashboard (General)')
# shot_dash_general
player_shot_dash_general = shot_dash_general[shot_dash_general['PLAYER'] == player]
# drop fgm cols
fgm_cols = [col for col in player_shot_dash_general.columns if 'FGM' in col]
player_shot_dash_general = player_shot_dash_general.drop(columns = fgm_cols)
#drop unnamed cols
unnamed_cols = [col for col in player_shot_dash_general.columns if 'Unnamed' in col]
player_shot_dash_general = player_shot_dash_general.drop(columns = unnamed_cols)


# drop some cols
cols_drop = ['TEAM', 'AGE', 'GP', 'G', 'FREQ']
player_shot_dash_general = player_shot_dash_general.drop(columns = cols_drop)

# sdg_numcols are all columns after index
sdg_numcols = ['FGA', 'FG%', 'EFG%', '2FG FREQ', '2FGA', '2FG%', '3FG FREQ', '3PM', '3PA', '3P%']
# turn all columns numeric
player_shot_dash_general[sdg_numcols] = player_shot_dash_general[sdg_numcols].apply(pd.to_numeric)

# add positions to shot_dash_general using 'PLAYER' and 'player' columns
shot_dash_general['position'] = shot_dash_general['PLAYER'].map(primary_positions.set_index('player')['primary_position_bbref'])

# get averages for player position
position_shot_dash_general = shot_dash_general[shot_dash_general['position'] == position]

# drop fgm cols -- we can figure that out through % and fga
fgm_cols = [col for col in position_shot_dash_general.columns if 'FGM' in col]
position_shot_dash_general = position_shot_dash_general.drop(columns = fgm_cols)

#drop unnamed cols
unnamed_cols = [col for col in position_shot_dash_general.columns if 'Unnamed' in col]
position_shot_dash_general = position_shot_dash_general.drop(columns = unnamed_cols)

# drop more cols
cols_drop = ['TEAM', 'AGE', 'GP', 'G', 'FREQ']
position_shot_dash_general = position_shot_dash_general.drop(columns = cols_drop)

# replace- with 0
position_shot_dash_general = position_shot_dash_general.replace('-', 0)


# turn numb_cols numeric
position_shot_dash_general[sdg_numcols] = position_shot_dash_general[sdg_numcols].apply(pd.to_numeric, errors = 'coerce')

# get averages, dropping any zero or null values
position_avg_shot_dash_general = position_shot_dash_general[sdg_numcols].astype(float).mean(axis = 0).to_frame().T
position_avg_shot_dash_general['PLAYER'] = 'Position Average'


# concat player_shot_dash_general and position_avg_shot_dash_general
position_shot_dash_general_comp = pd.concat([player_shot_dash_general, position_avg_shot_dash_general], axis = 0)
position_shot_dash_general_comp = position_shot_dash_general_comp.reset_index()


# check if player is in position_shot_dash_general. If not, add player_shot_dash_general to position_shot_dash_general
if player not in position_shot_dash_general['PLAYER'].values:
    position_shot_dash_general = pd.concat([position_shot_dash_general, player_shot_dash_general], axis = 0)

# get player percentile for each column
for col in sdg_numcols:
    position_shot_dash_general[col + '_percentile'] = position_shot_dash_general[col].rank(pct = True, method = 'first')

# multiple by 100
for col in position_shot_dash_general.columns:
    if 'percentile' in col:
        position_shot_dash_general[col] = position_shot_dash_general[col] * 100

# find player row
player_row = position_shot_dash_general[position_shot_dash_general['PLAYER'] == player]

player_percentile_cols = [col for col in player_row.columns if 'percentile' in col]

# cols should be index and player_percentile_cols
cols = ['PLAYER'] + player_percentile_cols
player_row = player_row[cols]

# concat player_row to position_shot_dash_general_comp
# rename player_row cols, getting rid of '_percentile'
player_row = player_row.rename(columns = {col: col.replace('_percentile', '') for col in player_row.columns})
# concat to position_shot_dash_general_comp

# rename player name to 'Position Percentile'
player_row['PLAYER'] = 'Position Percentile'
# drop index
player_row = player_row.reset_index()

position_shot_dash_general_comp = pd.concat([position_shot_dash_general_comp, player_row], axis = 0)
position_shot_dash_general_comp = position_shot_dash_general_comp.reset_index()
# drop index
position_shot_dash_general_comp = position_shot_dash_general_comp.drop(columns = ['index'])
# drop index and level_0, make PLAYER index
position_shot_dash_general_comp = position_shot_dash_general_comp.drop(columns = ['level_0']).set_index('PLAYER')

percentile_cols = [col for col in position_shot_dash_general_comp.columns if '%' in col]

# identify third row
percentile_row = position_shot_dash_general_comp.iloc[2]

no_third_row = position_shot_dash_general_comp.drop(index = percentile_row.name)
st.table(no_third_row.style.format('{:.1f}'))

# only_third_row is dropping the first two
only_third_row = position_shot_dash_general_comp.drop(index = no_third_row.index)

def color_code_percentile2(val):
    if val < 40:
        color = 'red'
    elif val < 45 and val > 40:
        color = 'orange'
    elif val < 55 and val > 45:
        color = 'white',
    elif val > 55 and val < 75:
        color = 'lightgreen'
    elif val > 75:
        color = 'green'
    else:
        color = 'white'
    # return highlight color (background)
    return 'background-color: %s' % color

# add third row
st.table(only_third_row.style.format('{:.1f}').applymap(color_code_percentile2))


# get player percentile for each column
for col in sdg_numcols:
    position_shot_dash_general[col + '_percentile'] = position_shot_dash_general[col].rank(pct = True, method = 'first')

# plot position FGA vs EFG% with plotly
filt_position_shot_dash_general = position_shot_dash_general[position_shot_dash_general['FGA'] > 10]
fig = px.scatter(filt_position_shot_dash_general, x = 'FGA', y = 'EFG%', hover_name = 'PLAYER', color_discrete_sequence = px.colors.qualitative.Dark24)
fig.update_layout(title = 'Field Goal Attempts vs Effective Field Goal % for Position (Min 10 FGA)')
# make plot bigger
fig.update_traces(marker = dict(size = 10))
# add player names
fig.add_annotation(x = position_shot_dash_general_comp['FGA'].values[0], y = position_shot_dash_general_comp['EFG%'].values[0], text = player, showarrow = True, arrowhead = 1)

# fig.add_annotation(x = max_FGA, y = position_shot_dash_general_comp['EFG%'].values[0], text = 'Position Average', showarrow = False)
st.plotly_chart(fig, use_container_width = True)

# plot position 3PA vs 3P% with plotly
filt_position_shot_dash_general = position_shot_dash_general[position_shot_dash_general['3PA'] > 3]
fig = px.scatter(filt_position_shot_dash_general, x = '3PA', y = '3P%', hover_name = 'PLAYER', color_discrete_sequence = px.colors.qualitative.Dark24)
fig.update_layout(title = '3 Point Attempts vs 3 Point % for Position (Min 3 3PA)')
fig.update_traces(marker = dict(size = 10))
# add player names
fig.add_annotation(x = position_shot_dash_general_comp['3PA'].values[0], y = position_shot_dash_general_comp['3P%'].values[0], text = player, showarrow = True, arrowhead = 1)
# get min 3PA
min_3PA = filt_position_shot_dash_general['3PA'].min()
# get max 3PA
max_3PA = filt_position_shot_dash_general['3PA'].max()
# add one to max_3PA
max_3PA = max_3PA + 1
# add average line
fig.add_shape(type = 'line', x0 = min_3PA, y0 = position_shot_dash_general['3P%'].mean(), x1 = max_3PA, y1 = position_shot_dash_general['3P%'].mean(), line = dict(color = 'black', dash = 'dash'))
# add average text
fig.add_annotation(x = max_3PA, y = position_shot_dash_general['3P%'].mean(), text = 'Average', showarrow = False)

st.plotly_chart(fig, use_container_width = True)

player_specific_shottype = 'https://www.nba.com/stats/player/' + str(player_nba_id)+'/shooting'
st.sidebar.markdown('[Specific Shot Types]('+ player_specific_shottype + ')')

st.sidebar.markdown('[NBA Stats Glossary](https://www.nba.com/stats/help/glossary)')

# 2 columns
col1, col2 = st.columns(2)


# plot shots taken vs points scored with a line of best fit in plotly
fig = px.scatter(player_gbg, x = 'fga', y = 'pts', trendline = 'ols', color = 'Home', hover_name = 'Date', color_discrete_sequence = px.colors.qualitative.Dark24)
fig.update_layout(title = 'Shots Taken vs Points Scored for ' + player)
fig.update_traces(marker = dict(size = 10))
# add r squared
z = np.polyfit(player_gbg['fga'], player_gbg['pts'], 1)
p = np.poly1d(z)
r_squared = r2_score(player_gbg['pts'], p(player_gbg['fga']))
fig.add_annotation(x = player_gbg['fga'].max(), y = player_gbg['pts'].max(), text = 'R Squared: ' + str(round(r_squared, 2)), showarrow = False)
col1.plotly_chart(fig, use_container_width = True)



# plot minutes played vs points scored with a line of best fit in plotly
fig = px.scatter(player_gbg, x = 'min', y = 'pts', trendline = 'ols', color = 'Home', hover_name = 'Date', color_discrete_sequence = px.colors.qualitative.Dark24)
fig.update_layout(title = 'Minutes Played vs Points Scored for ' + player)
fig.update_traces(marker = dict(size = 10))
# add r squared
z = np.polyfit(player_gbg['min'], player_gbg['pts'], 1)
p = np.poly1d(z)
r_squared = r2_score(player_gbg['pts'], p(player_gbg['min']))
fig.add_annotation(x = player_gbg['min'].max(), y = player_gbg['pts'].max(), text = 'R Squared: ' + str(round(r_squared, 2)), showarrow = False)
col2.plotly_chart(fig, use_container_width = True)


# make an interactive 3d scatter plot with plotly
fig = px.scatter_3d(player_gbg, x = 'fga', y = 'min', z = 'pts', color = 'season', hover_name = 'season', opacity = 0.5)
fig.update_layout(title = 'Shots Taken, Minutes Played, and Points Scored for ' + player)
# make the axis labels more readable
fig.update_layout(scene = dict(xaxis_title = 'Shots Taken', yaxis_title = 'Minutes Played', zaxis_title = 'Points Scored'))
col1.plotly_chart(fig, use_container_width = False)

# Make an interactive 3d scatter plot on efg%, 3p%, and fga with plotly
fig = px.scatter_3d(player_gbg, x = 'fga', y = '3p%', z = 'adv_efg%', color = 'season', hover_name = 'season', opacity = 0.5)
fig.update_layout(title = 'Effective Field Goal %, 3 Point %, and Shots Taken for ' + player)
# make the axis labels more readable
fig.update_layout(scene = dict(xaxis_title = 'Shots Taken', yaxis_title = '3 Point %', zaxis_title = 'Effective Field Goal %'))
col2.plotly_chart(fig, use_container_width = False)

# add div
st.markdown('---')
######################################################################################################

# Playtype breakdown
st.subheader('Playtype Breakdown')
playtype_folder = 'data/player/nba_com_playerdata/playtypes/'
pt_fold = os.listdir(playtype_folder)
# check if today's date is in file names
if 'pt_cut_' + str(today) + '_.csv' in pt_fold:
    # read in today's data
    pt_cut = pd.read_csv(playtype_folder + 'pt_cut_' + str(today) + '_.csv')
    pt_hand_off = pd.read_csv(playtype_folder + 'pt_hand_off_' + str(today) + '_.csv')
    pt_isolation = pd.read_csv(playtype_folder + 'pt_isolation_' + str(today) + '_.csv')
    pt_off_screen = pd.read_csv(playtype_folder + 'pt_off_screen_' + str(today) + '_.csv')
    pt_post_up = pd.read_csv(playtype_folder + 'pt_post_up_' + str(today) + '_.csv')
    pt_pr_ball_handler = pd.read_csv(playtype_folder + 'pt_pr_ball_handler_' + str(today) + '_.csv')
    pt_pr_roll_man = pd.read_csv(playtype_folder + 'pt_pr_roll_man_' + str(today) + '_.csv')
    pt_spot_up = pd.read_csv(playtype_folder + 'pt_spot_up_' + str(today) + '_.csv')
    pt_transition = pd.read_csv(playtype_folder + 'pt_transition_' + str(today) + '_.csv')
    pt_putbacks = pd.read_csv(playtype_folder + 'pt_putbacks_' + str(today) + '_.csv')
else:
    st.write('Todays Data Needs to be Collected')




player_iso = pt_isolation[pt_isolation['PLAYER'] == player]
if player_iso.empty:
    st.write('No Isolation Data for ' + player)
else:
    player_iso.index = ['Isolation Offense']

player_cut = pt_cut[pt_cut['PLAYER'] == player]
if player_cut.empty:
    st.write('No Cut Data for ' + player)
else:
    player_cut.index = ['Cut Offense']

player_hand_off = pt_hand_off[pt_hand_off['PLAYER'] == player]
if player_hand_off.empty:
    st.write('No Hand Off Data for ' + player)
else:
    player_hand_off.index = ['Hand Off Offense']

player_off_screen = pt_off_screen[pt_off_screen['PLAYER'] == player]
if player_off_screen.empty:
    st.write('No Off Screen Data for ' + player)
else:
    player_off_screen.index = ['Off Screen Offense']

player_post_up = pt_post_up[pt_post_up['PLAYER'] == player]
if player_post_up.empty:
    st.write('No Post Up Data for ' + player)
else:
    player_post_up.index = ['Post Up Offense']

player_pr_ball_handler = pt_pr_ball_handler[pt_pr_ball_handler['PLAYER'] == player]
if player_pr_ball_handler.empty:
    st.write('No Pick and Roll Ball Handler Data for ' + player)
else:
    player_pr_ball_handler.index = ['Pick and Roll Ball Handler Offense']

player_spot_up = pt_spot_up[pt_spot_up['PLAYER'] == player]
if player_spot_up.empty:
    st.write('No Spot Up Data for ' + player)
else:
    player_spot_up.index = ['Spot Up Offense']

player_transition = pt_transition[pt_transition['PLAYER'] == player]
if player_transition.empty:
    st.write('No Transition Data for ' + player)
else:
    player_transition.index = ['Transition Offense']

player_putbacks = pt_putbacks[pt_putbacks['PLAYER'] == player]
if player_putbacks.empty:
    st.write('No Putback Data for ' + player)
else:
    player_putbacks.index = ['Putback Offense']

# combine all the dataframes that have data
playtypes = pd.concat([player_iso, player_cut, player_hand_off, player_off_screen, player_post_up, player_pr_ball_handler, player_spot_up, player_transition, player_putbacks], axis = 0)

unnamed = [col for col in playtypes.columns if 'Unnamed' in col]
playtypes.drop(columns = unnamed, inplace = True)

st.table(playtypes.style.format('{:.2f}', subset= playtypes.columns[3:]).applymap(color_code_percentile2, subset = 'Percentile'))

# identify num cols (last 14 cols)
num_cols = player_iso.columns[-14:]


colz  = st.columns(2)
col1 = colz[0]
col2 = colz[1]
# plotly scatterplot of FREQ% vs Percentile, sized by Percentile
fig = px.scatter(playtypes, x = 'Percentile', y = 'Freq%', size = 'Percentile', color = playtypes.index, 
                                                            hover_name = playtypes.index, opacity = 0.5,
                                                            size_max = 40, width = 800, height = 600)
fig.update_layout(title = 'Frequency of Playtype vs Percentile for ' + player)
fig.update_layout(xaxis_title = 'Player NBA Percentile', yaxis_title = 'Frequency of Playtype')
# add annotations
for i in range(len(playtypes)):
    fig.add_annotation(x = playtypes['Percentile'][i], y = playtypes['Freq%'][i], text = playtypes.index[i])
col1.plotly_chart(fig, use_container_width = False)

# add playtype breakdwon donut chart with Freq%
fig = go.Figure(data = [go.Pie(labels = playtypes.index, values = playtypes['Freq%'], hole = 0.5)])
fig.update_layout(title = 'Playtype Breakdown for ' + player)
col2.plotly_chart(fig, use_container_width = False)

st.markdown('---')

st.subheader('Ball Handling & Assists')

col1, col2 = st.columns(2)

# calculate season averages by player from gbg_22, but keep the position column
position_season_averages = gbg_22.groupby(['trad_player', 'position']).mean().reset_index()

# filter by position
position_avg= position_season_averages[position_season_averages['position'] == position]
player_avg = position_season_averages[position_season_averages['trad_player'] == player]

# plot scatterplot of assist ratio to turnover%
fig = px.scatter(position_avg, x = 'adv_ast%', y = 'adv_ast/to', hover_name = 'trad_player', 
                                                opacity = 0.5, size_max = 40, width = 800, height = 600)
fig.update_layout(title = 'Assist Ratio vs Turnover% by Position')

# update x and y axis titles
x_title = 'Assist Percent (of Team total)'
y_title = 'Assist / Turnover Ratio'
fig.update_layout(xaxis_title = x_title, yaxis_title = y_title)

# add player scatter point to plot
fig.add_trace(go.Scatter(x = player_avg['adv_ast%'], y = player_avg['adv_ast/to'], mode = 'markers',
                        marker = dict(size = 20, color = 'red'), name = player))


col1.plotly_chart(fig, use_container_width = False)

# calculate percentiles for player in position

# check to see if player is in position_avg
if player not in position_avg['trad_player'].values:
    # add player_avg
    position_avg = position_avg.append(player_avg)

# make sure adv_ast/to is float
position_avg['adv_ast/to'] = position_avg['adv_ast/to'].astype(float)


position_avg['adv_ast%_percentile'] = position_avg['adv_ast%'].rank(pct = True)
position_avg['adv_ast/to_percentile'] = position_avg['adv_ast/to'].rank(pct = True)

player_ast_percent_percentile = position_avg[position_avg['trad_player'] == player]['adv_ast%_percentile'].values[0]
player_ast_to_percentile = position_avg[position_avg['trad_player'] == player]['adv_ast/to_percentile'].values[0]
player_filtered_avg = position_avg[position_avg['trad_player'] == player]

# add metrics
player_ast_percent = player_filtered_avg['adv_ast%'].values[0]
col2.metric('Assist Percent (Percent of Team Assists when on Floor)', value = player_ast_percent.round(1), delta = (str(player_ast_percent_percentile.round(2)*100) + ' percentile (Higher is better)'))
player_ast_to = player_filtered_avg['adv_ast/to'].values[0]
col2.metric('Assist / Turnover Ratio', value = player_ast_to.round(1), delta = (str(player_ast_to_percentile.round(2) *100) + ' percentile (Higher is better)'))
# find the column with 'to' and 'ratio' in it
to_ratio_col = [col for col in player_filtered_avg.columns if 'to' in col and 'ratio' in col][0]

# add percentile to position_avg
position_avg['adv_to ratio percentile'] = position_avg[to_ratio_col].rank(pct = True)
player_filtered_avg = position_avg[position_avg['trad_player'] == player]

advanced_to_ratio = player_filtered_avg[to_ratio_col].values[0]
col2.metric('Turnover Ratio (Average Player Turnovers per 100 Possessions)', value = advanced_to_ratio.round(1), delta = (str(player_filtered_avg['adv_to ratio percentile'].values[0].round(2) *100) + ' percentile (Lower is better)'))

position_avg['ast_percentile'] = position_avg['trad_ast'].rank(pct = True)
player_filtered_avg = position_avg[position_avg['trad_player'] == player]
player_ast = player_filtered_avg['trad_ast'].values[0]
col2.metric('Assists per Game', value = player_ast.round(1), delta = (str(player_filtered_avg['ast_percentile'].values[0].round(2) *100) + ' percentile (Higher is better)'))

st.markdown('---')

st.subheader('Last 10 Games')

# get last 10 games by using .head(10)
last_10 = player_gbg.head(10)

# drop Unnamed columns
unnamed = [col for col in last_10.columns if 'Unnamed' in col]
last_10.drop(columns = unnamed, inplace = True)

# get numeric cols
num_cols = last_10.columns[4:]
# make sure all num_cols are numeric
last_10[num_cols] = last_10[num_cols].apply(pd.to_numeric, errors = 'coerce')
# drop season_type column
last_10.drop(columns = ['season_type', 'adv_min', 'adv_season', 'Date'], inplace = True)
num_cols = last_10.columns[4:]

# add metrics, compare to season average
last_10_ppg = last_10['pts'].mean()
last_10_ppm = last_10['ppm'].mean()
last_10_3p = last_10['3pm'].sum() / last_10['3pa'].sum() *100
last_10_ast = last_10['ast'].mean()
last_10_reb = last_10['reb'].mean()
last_10_stl = last_10['stl'].mean()
last_10_reb = last_10['reb'].mean()
last_10_efg = last_10['adv_efg%'].mean()
last_10_ts = last_10['adv_ts%'].mean()
last_10_usg = last_10['adv_usg%'].mean()

player_gbg_22 = player_gbg[player_gbg['adv_season'] == 2022]

season_ppg = player_gbg_22['pts'].mean()
season_ppm = player_gbg_22['ppm'].mean()
season_3p = player_gbg_22['3pm'].sum() / player_gbg_22['3pa'].sum() *100
season_ast = player_gbg_22['ast'].mean()
season_reb = player_gbg_22['reb'].mean()
season_stl = player_gbg_22['stl'].mean()
season_reb = player_gbg_22['reb'].mean()
season_efg = player_gbg_22['adv_efg%'].mean()
season_ts = player_gbg_22['adv_ts%'].mean()
season_usg = player_gbg_22['adv_usg%'].mean()

# compare, using metrics
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
col1.metric(label = 'Points Per Game', value = round(last_10_ppg, 1), delta = round(last_10_ppg - season_ppg, 1))
col2.metric(label = 'Points Per Minute', value = round(last_10_ppm, 1), delta = round(last_10_ppm - season_ppm, 1))
col3.metric(label = '3P%', value = round(last_10_3p, 1), delta = round(last_10_3p - season_3p, 1))
col4.metric(label = 'AST', value = round(last_10_ast, 1), delta = round(last_10_ast - season_ast, 1))
col5.metric(label = 'REB', value = round(last_10_reb, 1), delta = round(last_10_reb - season_reb, 1))
col6.metric(label = 'STL', value = round(last_10_stl, 1), delta = round(last_10_stl - season_stl, 1))
col7.metric(label = 'eFG%', value = round(last_10_efg, 1), delta = round(last_10_efg - season_efg, 1))
col8.metric(label = 'TS%', value = round(last_10_ts, 1), delta = round(last_10_ts - season_ts, 1))
col9.metric(label = 'USG%', value = round(last_10_usg, 1), delta = round(last_10_usg - season_usg, 1))

# Display last 10 games
st.dataframe(last_10.style.format('{:.1f}', subset = num_cols))

st.write('Over the past 10 games, ' + player + ' has scored at a pace of ' + str(round(last_10['ppm'].mean(), 1)) + ' points per minute, ' + str(round(last_10['pts'].mean(), 1)) + ' points per game, while shooting ' + str(round(last_10['3p%'].mean(), 1)) + '% from three.')
st.write('He is averaging ' + str(round(last_10['ast'].mean(), 1)) + ' assists per game, ' + str(round(last_10['reb'].mean(), 1)) + ' rebounds per game, and ' + str(round(last_10['stl'].mean(), 1)) + ' steals per game.')
# Visualize PPM, Points, and 3P% over last 10 games
colz  = st.columns(3)
col1 = colz[0]
col2 = colz[1]
col3 = colz[2]

# PPM
fig = px.bar(last_10, x = 'game date', y = 'ppm', title = 'PPM over Last 10 Games', color = 'ppm', color_continuous_scale = 'RdBu')
fig.update_layout(xaxis_title = 'Date', yaxis_title = 'PPM')
# add player average
fig.add_hline(y = last_10['ppm'].mean(), line_dash = 'dash', line_color = 'red')
col1.plotly_chart(fig, use_container_width = False)

# Points
fig = px.bar(last_10, x = 'game date', y = 'pts', title = 'Points over Last 10 Games', color = 'pts', color_continuous_scale = 'RdBu')
fig.update_layout(xaxis_title = 'Date', yaxis_title = 'Points')
# add player average
fig.add_hline(y = last_10['pts'].mean(), line_dash = 'dash', line_color = 'red')
col2.plotly_chart(fig, use_container_width = False)

# 3P%
fig = px.bar(last_10, x = 'game date', y = '3p%', title = '3P% over Last 10 Games', color = '3p%', color_continuous_scale = 'RdBu')
fig.update_layout(xaxis_title = 'Date', yaxis_title = '3P%')
# add player average
fig.add_hline(y = last_10['3p%'].mean(), line_dash = 'dash', line_color = 'red')
col3.plotly_chart(fig, use_container_width = False)

col1.write('Usage')
# plot usage distribution over last 10
fig = px.violin(last_10, x = 'adv_usg%', title = 'Usage Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Usage %', yaxis_title = 'Frequency')
col1.plotly_chart(fig, use_container_width = False)

col2.write('Offensive Rating')
# plot offensive rating distribution over last 10
fig = px.violin(last_10, x = 'adv_offrtg', title = 'Offensive Rating Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Offensive Rating', yaxis_title = 'Frequency')
col2.plotly_chart(fig, use_container_width = False)

col3.write('Defensive Rating')
# plot defensive rating distribution over last 10
fig = px.violin(last_10, x = 'adv_defrtg', title = 'Defensive Rating Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Defensive Rating', yaxis_title = 'Frequency')
col3.plotly_chart(fig, use_container_width = False)




