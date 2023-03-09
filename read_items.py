#
#AUTHOR: KIAN ANKERSON
#DESC: HW2 CS 315
#RECOMMENDATION ALGORTHIMS
import numpy
import pandas
import pandas as pd
import numpy as np
import numpy.ma as ma

import time

def read_movies(file_name):
    try:
        movies = pd.read_csv(file_name)
        return movies
    except OSError:
        raise FileNotFoundError("Error reading File")


def read_ratings(file_name):
    print("Reading Ratings...")
    try:
        ratings = pd.read_csv(file_name)
        ratings.drop(['timestamp'], axis=1, inplace=True)
        # ratings = ratings.sort_values(by='movieId')
        # ratings.reset_index(drop=True, inplace= True)
        ratings.sort_values(by=['movieId'], inplace=True)
        print("Done Reading Ratings.")
        return ratings
    except OSError:
        raise FileNotFoundError("Error reading File")


def read_tags(file_name):
    try:
        tags = pd.read_csv(file_name)
        return tags
    except OSError:
        raise FileNotFoundError("Error reading File")


def read_links(file_name):
    try:
        links = pd.read_csv(file_name)
        return links
    except OSError:
        raise FileNotFoundError("Error reading File")




#the ratings are already sorted by movieId, therefore the lower index = lower movie id
def contsruct_item_profiles(ratings, num_users, num_movies):
    print("Constructing Item Profiles...")
    # ratings = pd.DataFrame()
    #movie_dict = dict()
    #movie_dict = pd.DataFrame([['1','0']], columns=['movieId', 'row_number_in_item_profiles'])
    matrix = np.zeros(shape=(num_movies, num_users), dtype=float)
    counter = 0
    past_movie = -1

    list_of_dicts = list()
    for index, row in ratings.iterrows():
        movie = int(row['movieId'])
        user = int(row['userId'])
        rating = row['rating']
        if movie != past_movie:
            #add it to movie_dict
            list_of_dicts.append({'movieId':str(movie), 'row_number_in_item_profiles':str(counter)})
            #movie_dict = movie_dict.append(temp_df, ignore_index=True)
            counter += 1
            past_movie = movie
        matrix[counter-1, user - 1] = rating
    matrix = ma.masked_where(matrix == 0, matrix)
    movie_dict = pd.DataFrame.from_dict(list_of_dicts)
    print("Done Constructing")
    return [matrix, movie_dict]


row_num = 0
row_num2 = 0


def normalize_all(item_profiles, norm_dict):
    print("Normalizing profiles...")

    def normalize(item_profile):
        global row_num
        # test = row_num
        mean = ma.mean(item_profile)
        item_profile = ma.subtract(item_profile, mean)
        norm_dict[row_num] = np.linalg.norm(item_profile)
        row_num += 1  # how to update row number
        return item_profile

    item_profiles = ma.apply_along_axis(normalize, 1, item_profiles)

    print("Done normalizing.")
    return item_profiles


# item profiles are already normalized, now just calculate similarities
def calc_sim_all(item_profiles, norm_dict, num_movies):
    print("Calculating Item Similarities...")
    sim_matrix = np.zeros(shape=(num_movies, num_movies))
    item_profiles[np.isnan(item_profiles)] = 0
    time1start = time.time()
    for x in range(0,num_movies):
    #for x in range(0, num_movies):
        x_norm = norm_dict[x]
        for x2 in range(x, num_movies):
        #for x2 in range(x, num_movies):
            x2_norm = norm_dict[x2]
            dot = ma.dot(item_profiles[x], item_profiles[x2])
            sim_matrix[x][x2] = dot / (x_norm * x2_norm)
    sim_matrix = sim_matrix + sim_matrix.T - np.diag(np.diag(sim_matrix))

    time1end = time.time()
    print(f"TIME 1: {time1end-time1start}")
    #^^Code Snippet I found that copies upper right triangular matrix to lower left
    #https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix
    print("Done calculating similarities")
    return sim_matrix


#the indexes are correlated to the movieIds, so the lower index = lower movie id
def compute_neighborhood_all(sim_matrix):
    # convert matrix to dataframe
    print("Computing Neighborhoods...")
    df = pandas.DataFrame(sim_matrix)
    big_dict = {}
    def compute_neighborhood(col):
        global row_num2
        column = pd.DataFrame(col)
        #set column name
        column.columns = ['ratings']
        #get row index as a column
        column.reset_index(inplace=True)
        column = column.loc[column['ratings'] < 1]
        #sort to get the top 5
        column.sort_values(by=['ratings', 'index'], ascending=[False, True], inplace=True)
        # keep only first 5
        column = column.head(5)
        col_label = "movie_index_" + str(row_num2)
        big_dict[col_label] = column['index'].values
        row_num2 += 1


    #df = df.iloc[0:10,0:10]
    df.apply(compute_neighborhood)
    top5 = pandas.DataFrame.from_dict(big_dict)
    yo = 0
    print("Done computing neighborhoods.")
    return top5

def estimate_ratings_all(item_profiles,neighborhoods, sim_matrix, num_movies, num_users):
    #check if we normalize, or use original ratings?
    print("Estimating Ratings...")
    def estimate_rating(movieIndex, userIndex, movie_pos):
        # get neighborhood
        sim_movies = neighborhoods[str('movie_index_' + str(movieIndex))].values
        numerator = denominator = 0
        for movie in sim_movies:
            other_movie_position = movie
            cur_rating = item_profiles[other_movie_position, userIndex]
            numerator += (sim_matrix[movie_pos, other_movie_position] * cur_rating)
            denominator += sim_matrix[movie_pos, other_movie_position]
        return numerator / denominator

    estimated_matrix = np.zeros(shape=(num_movies, num_users))
    item_profiles = ma.getdata(item_profiles)
    #go down every column of item profiles
    #for i in range(0,10):
    for u in range(0,num_users):
        #for j in range(0,10):
        for m in range(0,num_movies):
            #check is missing a rating
            if item_profiles[m,u] == 0:
                #get movie_id
                movieIndex = m
                userIndex = u
                estimated_matrix[m,u] = estimate_rating(movieIndex,userIndex, movie_pos=m)
    print("Done estimating ratings")
    return estimated_matrix

def get_top_5_recs(estimated_matrix):
    print("Getting top 5 recs...")
    df = pd.DataFrame(estimated_matrix)
    big_dict = {}
    global row_num
    row_num = 0
    def get5(col):
        global row_num
        column = pd.DataFrame(col)
        column.columns = ['ratings']
        # get row index as a column
        column.reset_index(inplace=True)
        # sort to get the top 5
        column.sort_values(by=['ratings', 'index'], ascending=[False, True], inplace=True)
        # keep only first 5
        column = column.head(5)
        col_label = "user_id_" + str(row_num + 1)
        big_dict[col_label] = column['index'].values
        row_num += 1

    df.apply(get5)
    top5_recs = pandas.DataFrame.from_dict(big_dict)
    yo = 0  # HERE combing dicts
    print("done getting top 5 recs")
    return top5_recs

def output(top5recs, movie_dict):
    print("Writing Output file...")
    global row_num
    row_num = 0
    try:
        outfile = open('output.txt', mode='w')
    except OSError:
        print("Error opening output file!")
        return
    def write_recs(data):
        global row_num
        user = row_num +1
        rec1 = movie_dict.query(f'row_number_in_item_profiles == "{data[0]}"')['movieId'].values[0]
        rec2 = movie_dict.query(f'row_number_in_item_profiles == "{data[1]}"')['movieId'].values[0]
        rec3 = movie_dict.query(f'row_number_in_item_profiles == "{data[2]}"')['movieId'].values[0]
        rec4 = movie_dict.query(f'row_number_in_item_profiles == "{data[3]}"')['movieId'].values[0]
        rec5 = movie_dict.query(f'row_number_in_item_profiles == "{data[4]}"')['movieId'].values[0]
        outfile.write(f"User-ID {user}\t{rec1}\t{rec2}\t{rec3}\t{rec4}\t{rec5}\n")
        row_num +=1

    top5recs.apply(write_recs)
    outfile.close()
    print("Done writing output file")

def main():
    num_users = 671
    num_movies = 9125
    norm_dict = dict()
    ratings = read_ratings('movie-lens-data/ratings.csv')
    results = contsruct_item_profiles(ratings=ratings,num_users=num_users,num_movies=num_movies)
    item_profiles = results[0]
    movie_dict = results[1]
    del results
    item_profiles = normalize_all(item_profiles, norm_dict)
    sim_matrix = calc_sim_all(item_profiles, norm_dict, num_movies=num_movies)
    numpy.savetxt("sim_matrix.csv", sim_matrix, delimiter=',')
    sim_matrix = np.genfromtxt('sim_matrix.csv', delimiter=',')

    neighborhoods = compute_neighborhood_all(sim_matrix)
    estimated_matrix = estimate_ratings_all(item_profiles,neighborhoods,sim_matrix,num_movies,num_users)
    top5recs = get_top_5_recs(estimated_matrix)
    output(top5recs, movie_dict)

    print("Done!")

if __name__ == "__main__":
    main()
