import pandas as pd
from sklearn.impute import SimpleImputer

def data_pipeline(file_name):
    ufc = pd.read_csv(f'data/{file_name}')

    red_names = ufc['RedFighter']
    blue_names = ufc['BlueFighter']

    # creating relevant feauters by taking the difference between fighters' stats
    ufc['draw_diff'] = (ufc['BlueDraws'] - ufc['RedDraws'])
    ufc['avg_sig_str_pct_diff'] = (ufc['BlueAvgSigStrPct'] - ufc['RedAvgSigStrPct'])
    ufc['avg_TD_pct_diff'] = (ufc['BlueAvgTDPct'] - ufc['RedAvgTDPct'])

    ufc['M_DEC_diff'] = (ufc['BlueWinsByDecisionMajority'] - ufc['RedWinsByDecisionMajority'])
    ufc['S_DEC_diff'] = (ufc['BlueWinsByDecisionSplit'] - ufc['RedWinsByDecisionSplit'])
    ufc['U_DEC_diff'] = (ufc['BlueWinsByDecisionUnanimous'] - ufc['RedWinsByDecisionUnanimous'])

    ufc['TKO_diff'] = (ufc['BlueWinsByTKODoctorStoppage'] - ufc['RedWinsByTKODoctorStoppage'])
    ufc['odds_diff'] = (ufc['BlueOdds'] - ufc['RedOdds'])
    ufc['ev_diff'] = (ufc['BlueExpectedValue'] - ufc['RedExpectedValue'])

    # dropping tons of irrelevant cols
    redundant = ['BlueOdds', 'RedOdds', 'BlueExpectedValue', 'RedExpectedValue', 'BlueCurrentLoseStreak', 
                'RedCurrentLoseStreak', 'BlueCurrentWinStreak', 'RedCurrentWinStreak', 'BlueLongestWinStreak', 
                'RedLongestWinStreak', 'BlueWins', 'RedWins', 'BlueLosses', 'RedLosses', 'BlueTotalRoundsFought',
                'RedTotalRoundsFought', 'BlueTotalTitleBouts', 'RedTotalTitleBouts', 'BlueWinsByKO',
                'RedWinsByKO', 'BlueWinsBySubmission', 'RedWinsBySubmission', 'BlueHeightCms', 'RedHeightCms',
                'BlueReachCms', 'RedReachCms', 'BlueAge', 'RedAge', 'BlueAvgSigStrLanded', 'RedAvgSigStrLanded',
                'BlueAvgSubAtt', 'RedAvgSubAtt', 'BlueAvgTDLanded', 'RedAvgTDLanded', 'BlueDraws', 'RedDraws',
                'BlueAvgSigStrPct', 'RedAvgSigStrPct', 'BlueAvgTDPct', 'RedAvgTDPct', 'BlueWinsByDecisionMajority',
                'RedWinsByDecisionMajority', 'BlueWinsByDecisionSplit', 'RedWinsByDecisionSplit', 
                'BlueWinsByDecisionUnanimous', 'RedWinsByDecisionUnanimous', 'BlueWinsByTKODoctorStoppage', 'RedWinsByTKODoctorStoppage']
    ufc.drop(redundant, axis = 1, inplace = True)

    drop = ['Date', 'Location', 'Country', 'WeightClass', 'Gender','NumberOfRounds', 'EmptyArena', 'Finish',
            'FinishDetails', 'FinishRound', 'FinishRoundTime', 'TotalFightTimeSecs', 'BlueWeightLbs', 'RedWeightLbs']
    ufc.drop(drop, axis = 1, inplace = True)

    # correcting one mispelled datapoint
    ufc['BlueStance'].loc[ufc['BlueStance'] == 'Switch '] = 'Switch'

    # encoding 'stance', 'betterrank', 'titlebout'
    ufc['BlueStance'] = [4 if stance == 'Orthodox' else 3 if stance == 'Southpaw' else 2 if stance == 'Switch' else 1 for stance in ufc['BlueStance']]
    ufc['RedStance'] = [4 if stance == 'Orthodox' else 3 if stance == 'Southpaw' else 2 if stance == 'Switch' else 1 for stance in ufc['RedStance']]
    ufc['BetterRank'] = [-1 if rank == 'Red' else 1 if rank == 'Blue' else 0 for rank in ufc['BetterRank']]
    ufc['TitleBout'] = [1 if title_bout else 0 for title_bout in ufc['TitleBout']]

    ufc['Stance_diff'] = (ufc['BlueStance'] - ufc['RedStance'])
    ufc.drop(['BlueStance', 'RedStance', 'Winner'], axis = 1, inplace = True)

    # dropping more unnecessary cols
    ufc.drop(ufc.loc[:, 'BMatchWCRank':'BPFPRank'], axis = 1, inplace = True)
    ufc.drop(['RedFighter', 'BlueFighter'], axis = 1, inplace = True)
    ufc.drop(['RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'], axis = 1, inplace = True)

    # data imputation
    impute = SimpleImputer(strategy = 'median')
    impute.fit(ufc[['avg_sig_str_pct_diff', 'avg_TD_pct_diff', 'odds_diff', 'ev_diff']])
    ufc[['avg_sig_str_pct_diff', 'avg_TD_pct_diff', 'odds_diff', 'ev_diff']] = impute.transform(ufc[['avg_sig_str_pct_diff', 'avg_TD_pct_diff', 'odds_diff', 'ev_diff']])

    return ufc, red_names, blue_names