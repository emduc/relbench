import os

import numpy as np
import pandas as pd
import pooch

from relbench.data import Database, RelBenchDataset, Table
from relbench.tasks.stackex import (
    BadgesTask,
    EngageTask,
    RelatedPostTask,
    UserCommentOnPostTask,
    UsersInteractTask,
    VotesTask,
)
from relbench.utils import unzip_processor


class StackExDataset(RelBenchDataset):
    name = "rel-stackex"
    # 2 years gap
    val_timestamp = pd.Timestamp("2019-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")
    max_eval_time_frames = 1
    task_cls_list = [
        EngageTask,
        VotesTask,
        BadgesTask,
        UserCommentOnPostTask,
        RelatedPostTask,
        UsersInteractTask,
    ]

    def __init__(
        self,
        *,
        process: bool = False,
    ):
        self.name = f"{self.name}"
        super().__init__(process=process)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        ## remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes", "DownVotes", "LastAccessDate"],
            inplace=True,
        )

        posts.drop(
            columns=[
                "ViewCount",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "CommunityOwnedDate",
                "ClosedDate",
                "LastEditDate",
                "LastActivityDate",
                "Score",
                "LastEditorDisplayName",
                "LastEditorUserId",
            ],
            inplace=True,
        )

        comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        ## change time column to pd timestamp series
        comments["CreationDate"] = pd.to_datetime(comments["CreationDate"])
        badges["Date"] = pd.to_datetime(badges["Date"])
        postLinks["CreationDate"] = pd.to_datetime(postLinks["CreationDate"])

        postHistory["CreationDate"] = pd.to_datetime(postHistory["CreationDate"])
        votes["CreationDate"] = pd.to_datetime(votes["CreationDate"])
        posts["CreationDate"] = pd.to_datetime(posts["CreationDate"])
        users["CreationDate"] = pd.to_datetime(users["CreationDate"])

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            fkey_col_to_pkey_table={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkey_col_to_pkey_table={
                "UserId": "users",
            },
            pkey_col="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "RelatedPostId": "posts",  ## is this allowed? two foreign keys into the same primary
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "ParentId": "posts",  # notice the self-reference
                "AcceptedAnswerId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        return Database(tables)
    
    def shardDataset(self, num_shards: int = 2): 
        r"""Shard the dataset horizontally to simulate distributed data.
        
        Datasets are stored in the cache."""
        
        db = self.make_db()
        
        shards = []
        
        users = db.table_dict["users"]
        user_shards = np.array_split(users.df, num_shards)   
        shards.append(
                {"name": "users", 
                 "df": user_shards, 
                 "fk_to_table": users.fkey_col_to_pkey_table}
                )
        
        votes = db.table_dict["votes"]  
        vote_shards = []
        remaining_votes = votes.df
        
        # TODO add asserts throughout the process, especially on sizes
        remaining_posts = self.retrieve_foreign_rows(shards[0]["df"], vote_shards, remaining_votes, foreignKey="UserId")
        # for user_shard in shards[0]["df"]:  
        #     user_df_col = user_shard[["Id"]]
        #     filtered_votes = remaining_votes.merge(user_df_col, left_on="UserId", right_on="Id", how="inner")
            
        #     filtered_votes.rename(columns={'Id_x': 'Id'}, inplace=True)
        #     filtered_votes.drop(columns='Id_y', inplace=True)
            
            # remaining_votes = remaining_votes.merge(filtered_votes, on='Id', how="left", indicator=True)
            # remaining_votes = remaining_votes[remaining_votes['_merge'] == 'left_only']
            # remaining_votes.drop(columns={'_merge', 'UserId_y', 'PostId_y', 'VoteTypeId_y', 'CreationDate_y'}, inplace=True)
            # remaining_votes.rename(columns={'UserId_x': 'UserId', 'PostId_x': 'PostId', 'VoteTypeId_x': 'VoteTypeId', 'CreationDate_x': 'CreationDate'}, inplace=True)

            # remaining_votes = remaining_votes[~remaining_votes["Id"].isin(filtered_votes["Id"])]
            # vote_shards.append(filtered_votes)
        
        # Filling the shards with the remaining votes 
        vote_shards = self.fill_shards(num_shards, votes, vote_shards, remaining_votes)            
        
        shards.append(
                {"name": "votes", 
                 "df": vote_shards, 
                 "fk_to_table": votes.fkey_col_to_pkey_table}
                )   
        
        # Posts table
        posts = db.table_dict["posts"]
        post_shards = []
        remaining_posts = posts.df
            
        remaining_posts = self.retrieve_foreign_rows(shards[0]["df"], post_shards, remaining_posts, foreignKey="OwnerUserId")
        remaining_posts = self.retrieve_foreign_rows(shards[1]["df"], post_shards, remaining_posts, OnKey="PostId", foreignKey="Id")

            
        # for vote_shard in shards[1]["df"]:
        #     vote_df_id = vote_shard[["PostId"]]
        #     filtered_posts = remaining_posts.merge(vote_df_id, left_on="Id", right_on="PostId", how="inner")
            
        #     # filtered_posts.rename(columns={'Id_x': 'Id'}, inplace=True)
        #     filtered_posts.drop(columns='PostId', inplace=True)
        #     remaining_posts = remaining_posts[~remaining_posts["Id"].isin(filtered_posts["Id"])]
            
        #     post_shards.append(filtered_posts)
            
            
        post_shards = self.fill_shards(num_shards, posts, post_shards, remaining_posts)            
        
        shards.append(
        {"name": "posts", 
            "df": post_shards, 
            "fk_to_table": posts.fkey_col_to_pkey_table}
        )   
        
        for shard in shards:
            print("size of shard table " + shard["name"])
            for i in range(num_shards):
                print(shard["df"][i].shape[0])
        
        return db

    def retrieve_foreign_rows(self, foreign_shards, current_shards, remaining_rows, OnKey="Id", foreignKey="Id"):
        for foreign_shard in foreign_shards:
            user_df_col = foreign_shard[[OnKey]]
            filtered_rows = remaining_rows.merge(user_df_col, left_on=foreignKey, right_on=OnKey, how="inner")
            
            if OnKey == "Id":
                filtered_rows.rename(columns={'Id_x': 'Id'}, inplace=True)
                filtered_rows.drop(columns='Id_y', inplace=True)
            else:
                filtered_rows.drop(columns=OnKey, inplace=True)
                
            remaining_rows = remaining_rows[~remaining_rows["Id"].isin(filtered_rows["Id"])]
            current_shards.append(filtered_rows)
        return remaining_rows
    


    def fill_shards(self, num_shards, table, shards, remainings):
        expected_batch_size = table.df.shape[0] / num_shards
        result = []
        for shard in shards:
            empty_slots = expected_batch_size - shard.shape[0]
            if empty_slots > 0:
                additional_slots = remainings.sample(n=min(int(empty_slots), int(remainings.shape[0])))
                shard = pd.concat([shard, additional_slots])
                remainings = remainings.drop(additional_slots.index)
            result.append(shard)
        return result
    
# main function to debug the shardDataset function
if __name__ == "__main__":
    dataset = StackExDataset(process=True)
    dataset.shardDataset(num_shards=2)