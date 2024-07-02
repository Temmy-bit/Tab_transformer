import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attention(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Encoder(nn.Module):
    def __init__(self, d_model,num_layers= 5 , n_heads = 1 , d_ff = 2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

    def get_params(self, deep=True):
        return {
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = self.forward()
        return self

class Scaling():
    """Treats data as series
    writing a function to apply this will 
    require input as a dataframe but apply as a pandas series"""

    def min_max_scaling(data):
        min_val = data.min()
        max_val = data.max()
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data
    
    def z_score_scaling(data):
        mean = data.mean()
        std_dev = data.std()
        scaled_data = (data - mean) / std_dev
        return scaled_data
    
    # def robust_scaling(data):
    #     median = data.median()
    #     q1 = data.quantile(0.25)
    #     q3 = data.quantile(0.75)
    #     iqr = q3 - q1
    #     scaled_data = (data - median) / iqr
    #     return scaled_data

    def robust_scaling(data):
        median = np.nanmedian(data)
        q1 = np.nanpercentile(data,0.25)
        q3 = np.nanpercentile(data,0.75)
        iqr = q3 - q1
        if np.all(np.isnan([iqr, median])):
            return data 
        iqr = np.where(iqr == 0, 10, iqr)
        scaled_data = (data - median) / iqr
        
        return scaled_data.fillna(0)


class Tab_Former(nn.Module):
    """
    A model Using Encoders On Categorical Features And Scaling On Numerical Features
      And Also Displayinng It's Relationships With T-sne Plot    
    """
    def __init__(self,train_df,target:str, id :str, test_df = None, random_seed= 24):
        super(Tab_Former, self).__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.random_seed = random_seed
        self.id = id
                

    def preprocess_data(self):
        # self.combined_df = pd.concat([self.train_df.drop, self.test_df], axis=0)
        missing=self.train_df.isnull().sum().sort_values(ascending=False)
        missing=missing.drop(missing[missing==0].index)
        df = pd.DataFrame(missing)
        self.label = self.train_df[self.target]
        self.combined_df = pd.concat([self.train_df,self.test_df],axis=0);
        self.combined_df.drop([self.id,self.target],inplace=True,axis=1)
        self.combined_df.drop(missing.keys(),inplace=True,axis=1)

        if self.test_df != None:
            self.test_id = self.test_df[self.id]
        
        return self.combined_df, self.label
       
    
    def cat_num_split(self,combined_df):

        # check the numbers of categorical features in train_df
        self.cat_cols = combined_df.select_dtypes(include='object').columns.tolist()
        cat_col_indices = list(combined_df.columns.get_loc(col) for col in self.cat_cols)
        # print(f"Number of categorical features: {len(cat_cols)}")

        # check the numbers of numeric features in train_df
        self.num_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols_indices = list(combined_df.columns.get_loc(col) for col in self.num_cols)
        # print(f"Number of numeric features: {len(numeric_cols)}")      
        
        
        return self.cat_cols, self.num_cols

    def feature_map(self):
        kill = []
        for i in self.cat_cols:
            kill.append(list(self.combined_df[i].unique()))
        # print(kill)
            
        self.feature_mapping = {}
        idx = 0

        # Iterate over each list of categories
        for category_list in kill:
        # Iterate over each category in the list
            for category in category_list:
                # Assign unique index to the category and update the index
                self.feature_mapping[category] = idx
                idx += 1
        
        return self.feature_mapping

    def label_encode(self):
        lc = LabelEncoder()
        for i in self.cat_cols:
            # print(i)
            self.combined_df[i] = lc.fit_transform(self.combined_df[i])
            # print(self.combined_df[self.combined_df[i]].head())
        self.label_encoded = self.combined_df[self.cat_cols]
        # print(self.label_encoded)
        return self.label_encoded
    
    def encode(self,encoder = None,cat_col_embedding = None):
        
        # print("Mapped Cat_col",mapped_cat_col_indices_values_df.head())1

        # Map the values in train_df to numerical indices using feature_map
        if cat_col_embedding == None:
            self.mapped_cat_col_indices_values_df = self.combined_df[self.cat_cols].applymap(lambda x: self.feature_mapping.get(x, x))


        else:
           self.mapped_cat_col_indices_values_df = self.label_encoded

        # print(self.mapped_cat_col_indices_values_df)
        if encoder:
            encoder = encoder
        else:
            encoder = Encoder(d_model=len(self.cat_cols))

        # Example input tensor
        src_input = self.mapped_cat_col_indices_values_df.values
        src_input = torch.Tensor(src_input)
        # print("Source Input",src_input)

        new = []
        enc_values = []

        torch.manual_seed(self.random_seed)
        for i in np.arange(len(src_input)):
            new.append(encoder(src_input[i].unsqueeze(dim=0)))


        for i in new:
            enc_values.append(i.squeeze().detach().numpy())

        enc_values = pd.DataFrame(enc_values)
        # print(enc_values.head())
        # encoded = enc_values.values
        # print(enc_values[:5])
        enc_values.columns = self.cat_cols
        # print(enc_values.head())
        # enc_values = enc_values

        return enc_values


    def preprocess(self, scaling,encoded_values):

        numeric_cols_indices_values_df = pd.DataFrame(self.combined_df[self.num_cols].values, columns=self.combined_df[self.num_cols].columns)
        # print("Continouns Variable\n",numeric_cols_indices_values_df.tail(2))

        # encoder = Encoder(d_model=len(self.cat_cols))  
        self.encoder = encoded_values
        # print(self.mapped_cat_col_indices_values_df.head())


        scale = numeric_cols_indices_values_df.apply(scaling)

        df = pd.concat([encoded_values,scale],axis=1)

        for i in df:
            if df[i].isnull().sum() == 0:
                continue
            else:
                df[i].fillna((df[i].mean()), inplace=True)

        print(df.isnull().sum())
        shape = self.train_df.shape[0]
        

        train = df[:shape] 
        test = df[shape:]

        return train, test
    
    def tsne_plot(self,encoded_values, values= 500):
        data_np = self.mapped_cat_col_indices_values_df.values[:values]

        # Perform dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2,random_state=self.random_seed)
        data_tsne = tsne.fit_transform(data_np)

        # torch.manual_seed(24)
        # with torch.no_grad():
        #     encoded_data = encoder(src_input[:100])

        # Convert the encoded data to numpy array
        encoded_data_np = encoded_values.values[:values]

        # Perform dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2,random_state=24)
        encoded_data_tsne = tsne.fit_transform(encoded_data_np)

        # Plot the 2D representations
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=np.arange(len(data_tsne)), cmap='plasma')
        plt.title('t-SNE 2D Representation of Original Data')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Data Point Index')

        plt.subplot(1, 2, 2)
        plt.scatter(encoded_data_tsne[:, 0], encoded_data_tsne[:, 1], c=np.arange(len(encoded_data_tsne)), cmap='plasma')
        plt.title('t-SNE 2D Representation')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Data Point Index')

        plt.tight_layout()
        plt.show()

    
def log_rmse(y_true, y_pred):
    # Apply the natural logarithm to the actual and predicted values
    log_y_true = np.log(y_true + 1)  # Adding 1 to avoid log(0) which is undefined
    log_y_pred = np.log(y_pred + 1)

    # Calculate the RMSE on the log scale
    rmse = np.sqrt(mean_squared_error(log_y_true, log_y_pred))
    return rmse

def run_model(model,train, target):

    X_train,X_test, y_train,y_test = train_test_split(train,target,test_size=0.25,random_state=24)
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_r2_score = r2_score(y_train,y_train_pred)
    model_mse = mean_squared_error(y_train,y_train_pred)
    # model_log_rmse = log_rmse(y_train,y_train_pred)
    

    model_test_r2_score = r2_score(y_test,y_test_pred)
    model_test_mse = mean_squared_error(y_test,y_test_pred)
    model_test_rmse = np.sqrt(model_test_mse)
    # log_test_rmse = log_rmse(y_test,y_test_pred)

    print("Model Performance For Traning Set")
    print("--"*5)
    print("r2_score: ", model_r2_score)
    print("mean squared error: ", model_mse)
    print("rmse: ", model_train_rmse)
    # print("Log RMSE: ", model_log_rmse)
    print("--"*5)

    print("Model Performance For Test Set")
    print("--"*5)
    print("r2_score: ", model_test_r2_score)
    print("mean squared error: ", model_test_mse)
    print("rmse: ", model_test_rmse)
    # print("Test Log RMSE: ", log_test_rmse)
    print("--"*5)
    print(model)
    print("__"*5)

    model_name = model.__repr__()

    # Check if the length of the model name is greater than 20
    if len(model_name) > 20:
        # Take the first ten letters of the model name
        model_name = model_name[:10]

    return {"Model Name" : model_name,
            "r2_score" : model_test_r2_score.round(4),
            "mean squared error" : model_test_mse.round(4),
            "rmse" : model_test_rmse.round(4)}
            # "Log RMSE" : log_test_rmse.round(4)}



