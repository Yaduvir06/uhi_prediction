!pip install scikit-learn
!pip install rioxarray
!pip install stackstac
!pip install planetary-computer
!pip install pystac-client
!pip install odc-stac
!pip install bayesian-optimization
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
import joblib
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# For Sentinel-2 data
import warnings
warnings.filterwarnings('ignore')
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load

# Install required packages if not already installed
try:
    import stackstac
except ImportError:
    !pip install stackstac

try:
    import planetary_computer
except ImportError:
    !pip install planetary-computer

try:
    import pystac_client
except ImportError:
    !pip install pystac-client

try:
    from odc.stac import stac_load
except ImportError:
    !pip install odc-stac

try:
    import rioxarray
except ImportError:
    !pip install rioxarray

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    !pip install bayesian-optimization

# Function to check and clean features
def check_and_clean_features(features_dict):
    """
    Check and clean features dictionary to ensure it contains no NaN, infinity, or extremely large values.
    """
    cleaned_dict = {}
    for key, value in features_dict.items():
        if value is None or not np.isfinite(value) or abs(value) > 1e6:
            cleaned_dict[key] = 0.0  # Replace with a safe default
        else:
            cleaned_dict[key] = value
    return cleaned_dict
# Function to extract Sentinel-2 features for a given lat/lon
def extract_sentinel_features(latitude, longitude, time_window="2021-06-01/2021-09-01"):
    """
    Extract Sentinel-2 features for a given latitude and longitude.
    Returns a dictionary of features or a dictionary of default values if extraction fails.
    """
    # Check if latitude or longitude is NaN and return default values if so
    if pd.isna(latitude) or pd.isna(longitude):
        print(f"Skipping invalid coordinates: {latitude}, {longitude}")
        return get_default_features()

    try:
        # Define a small bounding box around the point
        buffer = 0.03  # Smaller buffer for efficiency
        lower_left = (latitude - buffer, longitude - buffer)
        upper_right = (latitude + buffer, longitude + buffer)
        bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

        # Search for Sentinel-2 data
        stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = stac.search(
            bbox=bounds,
            datetime=time_window,
            collections=["sentinel-2-l2a"],
            query={"eo:cloud_cover": {"lt": 30}},
        )
        items = list(search.get_items())
        if len(items) == 0:
            print(f"No data found for coordinates: {latitude}, {longitude}")
            return get_default_features()

        # Load the data
        resolution = 10  # meters per pixel
        scale = resolution / 111320.0  # degrees per pixel for crs=4326
        data = stac_load(
            items,
            bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            crs="EPSG:4326",
            resolution=scale,
            chunks={"x": 1024, "y": 1024},  # Smaller chunks for memory efficiency
            dtype="uint16",
            patch_url=planetary_computer.sign,
            bbox=bounds
        )

        # Create median composite
        median = data.median(dim="time").compute()

        # Calculate indices
        ndbi = (median.B11 - median.B08) / (median.B11 + median.B08)
        ndvi = (median.B08 - median.B04) / (median.B08 + median.B04)
        ndwi = (median.B03 - median.B08) / (median.B03 + median.B08)

        # Temperature proxy
        temp_proxy = (median.B01 + median.B04 + median.B06 + median.B08) / 4

        # Get the values at the exact point
        x_coord = median["longitude"].sel(longitude=longitude, method="nearest").values
        y_coord = median["latitude"].sel(latitude=latitude, method="nearest").values

        # Extract pixel values
        pixel_ndbi = float(ndbi.sel(longitude=x_coord, latitude=y_coord, method="nearest").values)
        pixel_ndvi = float(ndvi.sel(longitude=x_coord, latitude=y_coord, method="nearest").values)
        pixel_ndwi = float(ndwi.sel(longitude=x_coord, latitude=y_coord, method="nearest").values)
        pixel_temp = float(temp_proxy.sel(longitude=x_coord, latitude=y_coord, method="nearest").values)

        # Calculate neighborhood statistics (spatial context without using lat/lon directly)
        ndbi_mean = float(ndbi.mean().values)
        ndbi_std = float(ndbi.std().values)
        ndvi_mean = float(ndvi.mean().values)
        ndvi_std = float(ndvi.std().values)
        ndwi_mean = float(ndwi.mean().values)
        ndwi_std = float(ndwi.std().values)
        temp_mean = float(temp_proxy.mean().values)
        temp_std = float(temp_proxy.std().values)

        # Additional band statistics that might correlate with UHI
        b01_mean = float(median.B01.mean().values)
        b04_mean = float(median.B04.mean().values)
        b12_mean = float(median.B12.mean().values)

        # Optional: calculate more complex features
        # Urban/Rural ratio in buffer area
        urban_mask = (ndbi > 0.0) & (ndvi < 0.25) & (ndwi < -0.1)
        urban_ratio = float(urban_mask.mean().values)

        # Return as dictionary
        features = {
            'ndbi': pixel_ndbi,
            'ndvi': pixel_ndvi,
            'ndwi': pixel_ndwi,
            'temp_proxy': pixel_temp,
            'ndbi_mean': ndbi_mean,
            'ndbi_std': ndbi_std,
            'ndvi_mean': ndvi_mean,
            'ndvi_std': ndvi_std,
            'ndwi_mean': ndwi_mean,
            'ndwi_std': ndwi_std,
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'b01_mean': b01_mean,
            'b04_mean': b04_mean,
            'b12_mean': b12_mean,
            'urban_ratio': urban_ratio
        }

        return check_and_clean_features(features)

    except Exception as e:
        print(f"Error processing coordinates {latitude}, {longitude}: {str(e)}")
        return get_default_features()

# Function to return default feature values when extraction fails
def get_default_features():
    """Return a dictionary of default values for features when extraction fails."""
    return {
        'ndbi': 0.0,
        'ndvi': 0.0,
        'ndwi': 0.0,
        'temp_proxy': 0.0,
        'ndbi_mean': 0.0,
        'ndbi_std': 0.0,
        'ndvi_mean': 0.0,
        'ndvi_std': 0.0,
        'ndwi_mean': 0.0,
        'ndwi_std': 0.0,
        'temp_mean': 0.0,
        'temp_std': 0.0,
        'b01_mean': 0.0,
        'b04_mean': 0.0,
        'b12_mean': 0.0,
        'urban_ratio': 0.0
    }

# Function to process training data and extract features
def process_training_data(uhi_data_path, weather_data_path, sample_size=None):
    """
    Process the training data, extract Sentinel-2 features, and merge with weather data.
    """
    print("Loading UHI and weather data...")
    # Load UHI training data
    uhi_df = pd.read_csv(uhi_data_path)

    # Drop rows with NaN coordinates
    original_size = len(uhi_df)
    uhi_df = uhi_df.dropna(subset=['Latitude', 'Longitude'])
    dropped_size = original_size - len(uhi_df)
    if dropped_size > 0:
        print(f"Dropped {dropped_size} rows with NaN coordinates")

    # Optional: Use a sample for faster development
    if sample_size and sample_size < len(uhi_df):
        uhi_df = uhi_df.sample(sample_size, random_state=42)

    # Load weather data
    try:
        xls = pd.ExcelFile(weather_data_path)
        bronx_weather = xls.parse("Bronx")
        manhattan_weather = xls.parse("Manhattan")

        # Convert date/time columns to datetime
        uhi_df["datetime"] = pd.to_datetime(uhi_df["datetime"], errors="coerce")
        bronx_weather["Date / Time"] = pd.to_datetime(bronx_weather["Date / Time"], errors="coerce")
        manhattan_weather["Date / Time"] = pd.to_datetime(manhattan_weather["Date / Time"], errors="coerce")

        # Combine weather data and rename datetime column
        weather_df = pd.concat([bronx_weather, manhattan_weather], ignore_index=True)
        weather_df.rename(columns={"Date / Time": "datetime"}, inplace=True)

        # Find the nearest weather record for each UHI record
        print("Matching UHI data with nearest weather records...")
        weather_timestamps = weather_df["datetime"].values.astype("datetime64[s]").astype(np.int64)
        uhi_timestamps = uhi_df["datetime"].values.astype("datetime64[s]").astype(np.int64)
        tree = cKDTree(weather_timestamps.reshape(-1, 1))
        _, nearest_indices = tree.query(uhi_timestamps.reshape(-1, 1), k=1)

        # Merge weather features with UHI data
        merged_df = uhi_df.copy()
        weather_features = weather_df.iloc[nearest_indices].reset_index(drop=True)
        merged_df = pd.concat([merged_df, weather_features.drop(columns=["datetime"])], axis=1)
    except Exception as e:
        print(f"Error processing weather data: {str(e)}")
        print("Continuing with UHI data only...")
        merged_df = uhi_df.copy()

    # Extract Sentinel-2 features for each point
    print("Extracting Sentinel-2 features (this may take some time)...")
    sentinel_features = []

    for idx, row in merged_df.iterrows():
        print(f"Processing point {idx+1}/{len(merged_df)}", end="\r")
        features = extract_sentinel_features(row["Latitude"], row["Longitude"])
        sentinel_features.append(features)

    print("\nConverting features to DataFrame...")
    # Convert list of dicts to DataFrame
    sentinel_df = pd.DataFrame(sentinel_features)

    # Combine all features
    final_df = pd.concat([merged_df, sentinel_df], axis=1)

    # Save the processed data
    final_df.to_csv("processed_uhi_data.csv", index=False)
    print("Processed data saved to processed_uhi_data.csv")

    return final_df

def prepare_features(df):
    """
    Prepare features for modeling by handling missing values and selecting relevant columns.
    """
    print("Preparing features for modeling...")

    # Create a copy to avoid modifying the original
    df = df.copy()

    # First check if UHI Index exists and handle missing or invalid values
    if "UHI Index" in df.columns:
        # Check for NaN or infinite values in UHI Index
        mask = np.isfinite(df["UHI Index"])
        if not mask.all():
            print(f"Found {(~mask).sum()} invalid values in UHI Index. Removing those rows.")
            df = df[mask].reset_index(drop=True)

        # Drop datetime and coordinate columns
        X = df.drop(columns=["UHI Index", "datetime", "Longitude", "Latitude"], errors='ignore')
        y = df["UHI Index"]

        # Print summary statistics for y to verify
        print(f"UHI Index - min: {y.min()}, max: {y.max()}, mean: {y.mean()}, has NaN: {y.isna().any()}")
    else:
        print("Warning: 'UHI Index' column not found in the DataFrame")
        # For submission data that doesn't have UHI Index
        X = df.drop(columns=["datetime", "Longitude", "Latitude"], errors='ignore')
        y = None

    # Clean column names
    X.columns = X.columns.astype(str).str.replace('[\\[\\]\\<]', '', regex=True)

    # Handle missing values in X
    print("Handling missing values in features...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Handle infinite values in X
    print("Handling infinite values in features...")
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)
    X_imputed = X_imputed.fillna(X_imputed.mean())

    return X_imputed, y
def optimize_xgboost(X_train, y_train):
    """
    Optimize XGBoost hyperparameters using Bayesian optimization.
    """
    print("Optimizing XGBoost hyperparameters...")

    # Make sure X_train and y_train have the same length and aligned indices
    if len(X_train) != len(y_train):
        print(f"WARNING: Length mismatch between X_train ({len(X_train)}) and y_train ({len(y_train)})")
        # Create new DataFrames with reset indices
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

    # Verify there are no NaN or infinite values
    X_train_np = X_train.values
    y_train_np = y_train.values

    # Create masks for valid rows
    valid_X = np.all(np.isfinite(X_train_np), axis=1)
    valid_y = np.isfinite(y_train_np)
    valid_rows = valid_X & valid_y

    if not np.all(valid_rows):
        print(f"WARNING: Found {(~valid_rows).sum()} rows with invalid values. Removing them.")
        X_train = X_train.iloc[valid_rows].reset_index(drop=True)
        y_train = y_train.iloc[valid_rows].reset_index(drop=True)
        print(f"Remaining training samples: {len(X_train)}")

    def xgb_cv(max_depth, learning_rate, subsample, colsample_bytree, gamma):
        try:
            model = xgb.XGBRegressor(
                max_depth=int(max_depth),
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                n_estimators=100,  # Reduced for optimization speed
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )

            cv_scores = cross_val_score(model, X_train, y_train, scoring='r2',
                                      cv=KFold(n_splits=3, shuffle=True, random_state=42))
            return cv_scores.mean()
        except Exception as e:
            print(f"Error in cross-validation: {str(e)}")
            return -1.0  # Return a bad score on error

    param_bounds = {
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.7, 1.0),
        'gamma': (0, 2)
    }

    try:
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds=param_bounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=2, n_iter=5)  # Reduced for faster optimization

        best_params = optimizer.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        print("Best hyperparameters found:", best_params)

        return best_params
    except Exception as e:
        print(f"Error in Bayesian optimization: {str(e)}")
        # Return default parameters if optimization fails
        return {
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0
        }

# Function to build and train models
def build_models(X_train, X_test, y_train, y_test, best_params):
    """
    Build and train XGBoost and stacking ensemble models.
    """
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"XGBoost - MAE: {mae_xgb:.4f}, MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("Top 10 features by importance:")
    print(feature_importance.head(10))

    # Build stacking ensemble
    print("Training stacking ensemble model...")
    try:
        estimators = [
            ('xgb', xgb_model),
            ('svr', SVR(kernel='rbf')),
            ('dt', DecisionTreeRegressor(random_state=42))
        ]

        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1
        )

        stacking_model.fit(X_train, y_train)

        # Evaluate stacking model
        y_pred_stack = stacking_model.predict(X_test)
        mae_stack = mean_absolute_error(y_test, y_pred_stack)
        mse_stack = mean_squared_error(y_test, y_pred_stack)
        r2_stack = r2_score(y_test, y_pred_stack)
        print(f"Stacking Ensemble - MAE: {mae_stack:.4f}, MSE: {mse_stack:.4f}, R²: {r2_stack:.4f}")

        # Save models
        joblib.dump(xgb_model, "UHI_XGBoost_Model.pkl")
        joblib.dump(stacking_model, "UHI_Stacking_Ensemble_Model.pkl")
        print("Models saved as UHI_XGBoost_Model.pkl and UHI_Stacking_Ensemble_Model.pkl")

        return xgb_model, stacking_model
    except Exception as e:
        print(f"Error building stacking ensemble: {str(e)}")
        print("Returning XGBoost model only")
        # Save XGBoost model
        joblib.dump(xgb_model, "UHI_XGBoost_Model.pkl")
        print("XGBoost model saved as UHI_XGBoost_Model.pkl")
        return xgb_model, None

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Function to build and train simpler models as fallback
def build_fallback_models(X_train, X_test, y_train, y_test):
    """
    Build and train simpler models that might handle problematic data better.
    """
    print("Training fallback models...")

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate RF model
    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"RandomForest - MAE: {mae_rf:.4f}, MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Evaluate GB model
    y_pred_gb = gb_model.predict(X_test)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    print(f"GradientBoosting - MAE: {mae_gb:.4f}, MSE: {mse_gb:.4f}, R²: {r2_gb:.4f}")

    # Select best model based on R²
    if r2_rf > r2_gb:
        best_model = rf_model
        print("RandomForest selected as best fallback model")
    else:
        best_model = gb_model
        print("GradientBoosting selected as best fallback model")

    # Save the best model
    model_name = "UHI_Fallback_Model.pkl"
    joblib.dump(best_model, model_name)
    print(f"Fallback model saved as {model_name}")

    return best_model
# Function to process submission data
def process_submission_data(submission_file, model):
    """
    Process submission data and make predictions.
    """
    print("Processing submission data...")
    submission_df = pd.read_csv(submission_file)

    # Drop rows with NaN coordinates
    original_size = len(submission_df)
    submission_df = submission_df.dropna(subset=['Latitude', 'Longitude'])
    dropped_size = original_size - len(submission_df)
    if dropped_size > 0:
        print(f"Dropped {dropped_size} rows with NaN coordinates in submission data")

    # Extract Sentinel-2 features for submission points
    submission_features = []
    for idx, row in submission_df.iterrows():
        print(f"Processing submission point {idx+1}/{len(submission_df)}", end="\r")
        features = extract_sentinel_features(row["Latitude"], row["Longitude"])
        submission_features.append(features)

    # Convert to DataFrame
    submission_sentinel_df = pd.DataFrame(submission_features)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    submission_sentinel_df = pd.DataFrame(
        imputer.fit_transform(submission_sentinel_df),
        columns=submission_sentinel_df.columns
    )

    # Handle infinite values
    submission_sentinel_df = submission_sentinel_df.replace([np.inf, -np.inf], np.nan)
    submission_sentinel_df = submission_sentinel_df.fillna(submission_sentinel_df.mean())

    # Make predictions
    print("\nMaking predictions...")
    try:
        predictions = model.predict(submission_sentinel_df)

        # Add predictions to submission DataFrame
        submission_df["Predicted_UHI_Index"] = predictions

        # Save predictions
        output_file = "UHI_Predictions.csv"
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved as {output_file}")

        return submission_df
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        # Try a simpler approach - mean UHI index
        mean_uhi = 0.0  # Will be replaced if we have training data
        try:
            mean_uhi = np.mean(y)
        except:
            pass
        submission_df["Predicted_UHI_Index"] = mean_uhi
        output_file = "UHI_Predictions_Fallback.csv"
        submission_df.to_csv(output_file, index=False)
        print(f"Fallback predictions saved as {output_file}")
        return submission_df

# Alternative approach: Use KMeans clustering (from the second code)
def process_with_kmeans(uhi_data_path, submission_file):
    """
    Process the data using KMeans clustering as in the second code.
    """
    print("Processing with KMeans clustering approach...")

    # Load UHI training data
    uhi_df = pd.read_csv(uhi_data_path)

    # Drop rows with NaN coordinates
    uhi_df = uhi_df.dropna(subset=['Latitude', 'Longitude'])

    # Apply KMeans clustering on Longitude and Latitude
    num_clusters = 300
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    uhi_df["Cluster_Label"] = kmeans.fit_predict(uhi_df[["Longitude", "Latitude"]])

    # Create a cluster mapping
    cluster_mapping = uhi_df.groupby("Cluster_Label")["UHI Index"].mean().to_dict()
    joblib.dump(cluster_mapping, "Cluster_Mapping_Model.pkl")
    print("Cluster mapping model saved as Cluster_Mapping_Model.pkl")

    # Load submission data
    submission_df = pd.read_csv(submission_file)

    # Drop rows with NaN coordinates
    submission_df = submission_df.dropna(subset=['Latitude', 'Longitude'])

    # Assign clusters to submission data
    submission_df["Cluster_Label"] = kmeans.predict(submission_df[["Longitude", "Latitude"]])

    # Predict UHI index using cluster mapping
    submission_df["Predicted_UHI_Index"] = submission_df["Cluster_Label"].map(cluster_mapping)

    # Fill any NaN predictions with the mean UHI index
    mean_uhi = uhi_df["UHI Index"].mean()
    submission_df["Predicted_UHI_Index"] = submission_df["Predicted_UHI_Index"].fillna(mean_uhi)

    # Save predictions
    output_file = "UHI_Predictions_KMeans.csv"
    submission_df[["Longitude", "Latitude", "Predicted_UHI_Index"]].to_csv(output_file, index=False)
    print(f"KMeans predictions saved as {output_file}")

    return submission_df

# Main function to run the complete pipeline
# In the main function:
def main():
    try:
        # File paths
        uhi_data_path = "/content/Training_data_uhi_index_2025-02-18.csv"
        weather_data_path = "/content/NY_Mesonet_Weather.xlsx"
        submission_file = "/content/Submission_template_UHI2025-v2.csv"

        # Process training data (increase sample size for better accuracy)
        sample_size = 500  # Increased from 100
        processed_data = process_training_data(uhi_data_path, weather_data_path, sample_size)

        # Prepare data for modeling
        X, y = prepare_features(processed_data)

        # Reset indices before splitting to ensure alignment
        X = X.reset_index(drop=True)
        if y is not None:
            y = y.reset_index(drop=True)

            # Check if we have valid data for modeling
            if len(X) > 0 and len(y) > 0:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Reset indices after splitting
                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

                # Try multiple modeling approaches
                models = []
                scores = []

                # Continue with model training...
        # Try XGBoost approach first
        try:
            print("\n=== Trying XGBoost approach ===")
            best_params = optimize_xgboost(X_train, y_train)
            xgb_model, stacking_model = build_models(X_train, X_test, y_train, y_test, best_params)

            # Evaluate models
            if xgb_model is not None:
                y_pred_xgb = xgb_model.predict(X_test)
                r2_xgb = r2_score(y_test, y_pred_xgb)
                models.append(xgb_model)
                scores.append(r2_xgb)
                print(f"XGBoost R² score: {r2_xgb:.4f}")

            if stacking_model is not None:
                y_pred_stack = stacking_model.predict(X_test)
                r2_stack = r2_score(y_test, y_pred_stack)
                models.append(stacking_model)
                scores.append(r2_stack)
                print(f"Stacking Ensemble R² score: {r2_stack:.4f}")
        except Exception as e:
            print(f"Error in XGBoost approach: {str(e)}")

        # Try fallback models
        try:
            print("\n=== Trying fallback models ===")
            fallback_model = build_fallback_models(X_train, X_test, y_train, y_test)
            if fallback_model is not None:
                y_pred_fallback = fallback_model.predict(X_test)
                r2_fallback = r2_score(y_test, y_pred_fallback)
                models.append(fallback_model)
                scores.append(r2_fallback)
                print(f"Fallback model R² score: {r2_fallback:.4f}")
        except Exception as e:
            print(f"Error in fallback approach: {str(e)}")

        # Try KMeans clustering approach
        try:
            print("\n=== Trying KMeans clustering approach ===")
            # Process with KMeans
            kmeans_results = process_with_kmeans(uhi_data_path, submission_file)
            print("KMeans approach completed")
        except Exception as e:
            print(f"Error in KMeans approach: {str(e)}")

        # Select best model for predictions
        if models and scores:
            best_index = np.argmax(scores)
            best_model = models[best_index]
            best_score = scores[best_index]
            print(f"\nBest model selected with R² score: {best_score:.4f}")

            # Process submission with best model
            submission_df = process_submission_data(submission_file, best_model)
            print("Predictions completed with best model!")
        else:
            print("\nNo successful model training. Using KMeans predictions.")

        print("\nPipeline completed!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Please check your input data and try again.")
if __name__ == "__main__":
    main()