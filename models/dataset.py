import csv
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class CERTFeatureEngineer:
    """
    Transforms raw CERT CSV files into per-user, per-day feature vectors.
    
    WHY per-day? Because a single event is noisy and meaningless in isolation.
    A day aggregates behavior into a stable, comparable unit. Most insider threat
    research uses daily or weekly aggregation for this reason.
    
    WHY these features? Each feature captures a different dimension of behavioral
    risk. We want to encode: WHEN the person works, HOW MUCH they access,
    and WHAT type of activity they do.
    """
    
    TIMESTAMP_FORMAT = '%m/%d/%Y %H:%M:%S'

    def __init__(
        self,
        data_dir: str,
        dataset_release: str = '4.2',
        chunksize: int = 250_000,
        max_rows_per_file: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_release = str(dataset_release)
        self.chunksize = chunksize
        self.max_rows_per_file = max_rows_per_file
        self.scaler = StandardScaler()
        self.user_baselines = {}  # stores per-user mean/std for normalization
        
    def load_raw_data(self):
        """Validate activity files and load the matching ground truth."""
        print("Loading raw CERT data...")

        required_files = [
            self.data_dir / f'{name}.csv'
            for name in ['logon', 'device', 'file', 'email', 'http']
        ]
        missing_files = [path for path in required_files if not path.is_file()]
        if missing_files:
            missing = ', '.join(str(path) for path in missing_files)
            raise FileNotFoundError(f'Missing CERT activity files: {missing}')

        answers_path = self.data_dir / 'answers' / 'insiders.csv'
        insider_columns = ['dataset', 'scenario', 'details', 'user', 'start', 'end']

        if answers_path.is_file():
            insiders = pd.read_csv(answers_path)
            missing_columns = set(insider_columns) - set(insiders.columns)
            if missing_columns:
                raise ValueError(
                    f'{answers_path} is missing columns: '
                    f'{sorted(missing_columns)}'
                )

            release = pd.to_numeric(insiders['dataset'], errors='coerce')
            target_release = float(self.dataset_release)
            insiders = insiders[np.isclose(release, target_release)].copy()
            insiders['start'] = pd.to_datetime(
                insiders['start'],
                format=self.TIMESTAMP_FORMAT,
                errors='coerce',
            )
            insiders['end'] = pd.to_datetime(
                insiders['end'],
                format=self.TIMESTAMP_FORMAT,
                errors='coerce',
            )
            if insiders[['start', 'end']].isna().any().any():
                raise ValueError(
                    f'Could not parse all timestamps in {answers_path}'
                )
            self.insiders = insiders
            self.malicious_user_days = set()
            answers_dir = answers_path.parent
            for incident in self.insiders.itertuples(index=False):
                details_path = (
                    answers_dir / f'r{self.dataset_release}-{incident.scenario}' / str(incident.details).strip()
                )
                if details_path.is_file():
                    with details_path.open(
                        newline='', encoding='utf-8', errors='replace'
                    ) as details_file:
                        for record in csv.reader(details_file):
                            if len(record) < 4 or record[3] != incident.user:
                                continue
                            try:
                                event_time = datetime.strptime(
                                    record[2], self.TIMESTAMP_FORMAT
                                )
                            except ValueError:
                                continue
                            self.malicious_user_days.add(
                                (incident.user, pd.Timestamp(event_time).normalize())
                            )
                else:
                    for date in pd.date_range(
                        incident.start.normalize(),
                        incident.end.normalize(),
                        freq='D',
                    ):
                        self.malicious_user_days.add((incident.user, date))

            print(
                f'Loaded {len(self.insiders)} ground-truth incidents for '
                f'CERT release {self.dataset_release} '
                f'({len(self.malicious_user_days)} malicious user-days).'
            )
        else:
            self.insiders = pd.DataFrame(columns=insider_columns)
            self.malicious_user_days = set()
            print(
                f'Ground truth not found at {answers_path}. '
                'Continuing with unlabeled data.'
            )

    def _iter_activity_chunks(self, filename: str, usecols: list[str]):
        """Read selected columns and derive date/time fields in bounded memory."""
        path = self.data_dir / filename
        reader = pd.read_csv(
            path,
            usecols=usecols,
            chunksize=self.chunksize,
            nrows=self.max_rows_per_file,
        )
        rows_processed = 0

        for chunk in reader:
            timestamps = pd.to_datetime(
                chunk.pop('date'),
                format=self.TIMESTAMP_FORMAT,
                errors='coerce',
            )
            valid = timestamps.notna() & chunk['user'].notna()
            if not valid.all():
                chunk = chunk.loc[valid].copy()
                timestamps = timestamps.loc[valid]

            chunk['date'] = timestamps.dt.normalize()
            chunk['hour'] = timestamps.dt.hour.astype(np.int8)
            chunk['is_weekend'] = timestamps.dt.dayofweek.ge(5)
            rows_processed += len(chunk)
            yield chunk

        print(f'  {filename}: processed {rows_processed:,} rows')

    @staticmethod
    def _combine_parts(
        parts: list[pd.DataFrame],
        aggregations: dict[str, str],
    ) -> pd.DataFrame:
        if not parts:
            empty_index = pd.MultiIndex.from_arrays(
                [[], []], names=['user', 'date']
            )
            return pd.DataFrame(columns=list(aggregations), index=empty_index)

        combined = pd.concat(parts)
        return combined.groupby(
            ['user', 'date'], observed=True
        ).agg(aggregations)

    def _aggregate_logon_features(self):
        user_dates = []
        logon_parts = []
        logoff_parts = []

        for chunk in self._iter_activity_chunks(
            'logon.csv', ['date', 'user', 'activity']
        ):
            user_dates.append(chunk[['user', 'date']].drop_duplicates())

            logons = chunk[chunk['activity'].eq('Logon')].copy()
            if not logons.empty:
                logons['after_hours_logons'] = (
                    logons['hour'].lt(7) | logons['hour'].ge(20)
                ).astype(np.int8)
                logons['weekend_logons'] = logons['is_weekend'].astype(np.int8)
                partial = logons.groupby(
                    ['user', 'date'], observed=True
                ).agg(
                    logon_count=('activity', 'size'),
                    after_hours_logons=('after_hours_logons', 'sum'),
                    weekend_logons=('weekend_logons', 'sum'),
                    first_logon_hour=('hour', 'min'),
                )
                logon_parts.append(partial)

            logoffs = chunk[chunk['activity'].eq('Logoff')]
            if not logoffs.empty:
                partial = logoffs.groupby(
                    ['user', 'date'], observed=True
                ).agg(last_logoff_hour=('hour', 'max'))
                logoff_parts.append(partial)

        user_dates = pd.concat(user_dates, ignore_index=True).drop_duplicates()
        users = np.sort(user_dates['user'].unique())
        dates = np.sort(user_dates['date'].unique())
        full_index = pd.MultiIndex.from_product(
            [users, dates], names=['user', 'date']
        )

        logon_features = self._combine_parts(
            logon_parts,
            {
                'logon_count': 'sum',
                'after_hours_logons': 'sum',
                'weekend_logons': 'sum',
                'first_logon_hour': 'min',
            },
        )
        logoff_features = self._combine_parts(
            logoff_parts, {'last_logoff_hour': 'max'}
        )

        features = pd.DataFrame(index=full_index)
        features = features.join(logon_features).join(logoff_features)
        features[
            ['logon_count', 'after_hours_logons', 'weekend_logons']
        ] = features[
            ['logon_count', 'after_hours_logons', 'weekend_logons']
        ].fillna(0)
        features['first_logon_hour'] = features['first_logon_hour'].fillna(9)
        features['last_logoff_hour'] = features['last_logoff_hour'].fillna(17)
        return features

    def _aggregate_device_features(self):
        parts = []
        for chunk in self._iter_activity_chunks(
            'device.csv', ['date', 'user', 'activity']
        ):
            connects = chunk[chunk['activity'].eq('Connect')].copy()
            if connects.empty:
                continue
            connects['usb_after_hours'] = (
                connects['hour'].lt(7) | connects['hour'].ge(20)
            ).astype(np.int8)
            parts.append(
                connects.groupby(['user', 'date'], observed=True).agg(
                    usb_connect_count=('activity', 'size'),
                    usb_after_hours=('usb_after_hours', 'sum'),
                )
            )

        return self._combine_parts(
            parts,
            {
                'usb_connect_count': 'sum',
                'usb_after_hours': 'sum',
            },
        )

    def _aggregate_file_features(self):
        count_parts = []
        unique_files = []

        for chunk in self._iter_activity_chunks(
            'file.csv', ['date', 'user', 'filename']
        ):
            chunk['file_after_hours'] = (
                chunk['hour'].lt(7) | chunk['hour'].ge(20)
            ).astype(np.int8)
            chunk['exe_access_count'] = (
                chunk['filename'].str.lower().str.endswith('.exe', na=False)
            ).astype(np.int8)
            count_parts.append(
                chunk.groupby(['user', 'date'], observed=True).agg(
                    file_access_count=('filename', 'size'),
                    file_after_hours=('file_after_hours', 'sum'),
                    exe_access_count=('exe_access_count', 'sum'),
                )
            )
            unique_files.append(
                chunk[['user', 'date', 'filename']]
                .dropna(subset=['filename'])
                .drop_duplicates()
            )

        features = self._combine_parts(
            count_parts,
            {
                'file_access_count': 'sum',
                'file_after_hours': 'sum',
                'exe_access_count': 'sum',
            },
        )
        filenames = pd.concat(unique_files, ignore_index=True).drop_duplicates()
        filename_counts = filenames.groupby(
            ['user', 'date'], observed=True
        ).size().rename('unique_files')
        return features.join(filename_counts, how='outer')

    def _aggregate_email_features(self):
        parts = []
        for chunk in self._iter_activity_chunks(
            'email.csv',
            ['date', 'user', 'to', 'size', 'attachments'],
        ):
            chunk['external_email_count'] = (
                ~chunk['to'].str.contains('dtaa.com', case=False, na=False)
            ).astype(np.int8)
            chunk['emails_with_attachments'] = chunk['attachments'].gt(0).astype(
                np.int8
            )
            chunk['email_size_sum'] = pd.to_numeric(
                chunk['size'], errors='coerce'
            ).fillna(0)
            parts.append(
                chunk.groupby(['user', 'date'], observed=True).agg(
                    email_count=('user', 'size'),
                    external_email_count=('external_email_count', 'sum'),
                    emails_with_attachments=('emails_with_attachments', 'sum'),
                    email_size_sum=('email_size_sum', 'sum'),
                )
            )

        features = self._combine_parts(
            parts,
            {
                'email_count': 'sum',
                'external_email_count': 'sum',
                'emails_with_attachments': 'sum',
                'email_size_sum': 'sum',
            },
        )
        features['avg_email_size'] = (
            features['email_size_sum']
            .div(features['email_count'].replace(0, np.nan))
            .fillna(0)
        )
        return features.drop(columns='email_size_sum')

    def _aggregate_http_features(self):
        parts = []
        job_pattern = '|'.join(
            re.escape(keyword)
            for keyword in [
                'linkedin',
                'monster',
                'careerbuilder',
                'indeed',
                'glassdoor',
                'jobs',
            ]
        )
        cloud_pattern = '|'.join(
            re.escape(keyword)
            for keyword in [
                'dropbox',
                'drive.google',
                'onedrive',
                'box.com',
                'wetransfer',
            ]
        )

        for chunk in self._iter_activity_chunks(
            'http.csv', ['date', 'user', 'url']
        ):
            chunk['job_site_visits'] = chunk['url'].str.contains(
                job_pattern, case=False, na=False
            ).astype(np.int8)
            chunk['cloud_storage_visits'] = chunk['url'].str.contains(
                cloud_pattern, case=False, na=False
            ).astype(np.int8)
            parts.append(
                chunk.groupby(['user', 'date'], observed=True).agg(
                    http_count=('user', 'size'),
                    job_site_visits=('job_site_visits', 'sum'),
                    cloud_storage_visits=('cloud_storage_visits', 'sum'),
                )
            )

        return self._combine_parts(
            parts,
            {
                'http_count': 'sum',
                'job_site_visits': 'sum',
                'cloud_storage_visits': 'sum',
            },
        )
    
    def _extract_logon_features(self, user_df: pd.DataFrame) -> pd.Series:
        """
        WHY these logon features?
        
        - after_hours_logons: The single strongest signal in CERT data.
          Insiders often act at night when no one is watching.
        - session_count: Unusually high logins can mean account sharing
          or scripted access.
        - weekend_logons: Similar to after-hours — legitimate users rarely
          work weekends, especially late at night.
        - first_logon_hour: Did this person start unusually early or late?
          Captures the "sneak in before everyone arrives" pattern.
        """
        if user_df.empty:
            return pd.Series({
                'logon_count': 0,
                'after_hours_logons': 0,
                'weekend_logons': 0,
                'first_logon_hour': 9,  # assume normal 9am if no data
                'last_logoff_hour': 17,
            })
        
        logons = user_df[user_df['activity'] == 'Logon']
        
        return pd.Series({
            'logon_count': len(logons),
            # After hours = before 7am or after 8pm
            'after_hours_logons': ((logons['hour'] < 7) | 
                                    (logons['hour'] >= 20)).sum(),
            'weekend_logons': logons['is_weekend'].sum(),
            'first_logon_hour': logons['hour'].min() if len(logons) > 0 else 9,
            'last_logoff_hour': user_df[user_df['activity'] == 'Logoff']['hour'].max() 
                                if len(user_df) > 0 else 17,
        })
    
    def _extract_device_features(self, user_df: pd.DataFrame) -> pd.Series:
        """
        WHY these device (USB) features?
        
        USB insertions are a primary exfiltration vector. Most employees
        plug in USB drives rarely — maybe once a week, maybe never.
        A sudden spike is a huge red flag.
        
        - usb_connect_count: Raw count of USB insertions that day
        - usb_after_hours: USB activity at unusual hours is doubly suspicious
        """
        if user_df.empty:
            return pd.Series({'usb_connect_count': 0, 
                              'usb_after_hours': 0})
        
        connects = user_df[user_df['activity'] == 'Connect']
        return pd.Series({
            'usb_connect_count': len(connects),
            'usb_after_hours': ((connects['hour'] < 7) | 
                                 (connects['hour'] >= 20)).sum(),
        })
    
    def _extract_file_features(self, user_df: pd.DataFrame) -> pd.Series:
        """
        WHY these file features?
        
        File access volume is the clearest exfiltration signal. An insider
        preparing to leave will often bulk-copy files. We capture:
        
        - file_access_count: Raw volume. This alone is a strong feature.
        - unique_files: Many copies of same file = scripted copying
        - file_access_after_hours: Timing matters as much as volume
        - exe_access_count: Accessing executables is unusual for most users
          and could indicate reconnaissance or tool staging
        """
        if user_df.empty:
            return pd.Series({
                'file_access_count': 0,
                'unique_files': 0,
                'file_after_hours': 0,
                'exe_access_count': 0,
            })
        
        return pd.Series({
            'file_access_count': len(user_df),
            'unique_files': user_df['filename'].nunique() 
                            if 'filename' in user_df.columns else 0,
            'file_after_hours': ((user_df['hour'] < 7) | 
                                  (user_df['hour'] >= 20)).sum(),
            # .exe files accessed — unusual for non-IT users
            'exe_access_count': user_df['filename'].str.endswith('.exe').sum()
                                 if 'filename' in user_df.columns else 0,
        })
    
    def _extract_email_features(self, user_df: pd.DataFrame) -> pd.Series:
        """
        WHY these email features?
        
        Email is the most common exfiltration channel. Key signals:
        
        - email_count: Volume. Spikes before resignation are a known pattern.
        - external_email_count: Sending to outside the org is the red flag.
          Most legitimate work email stays internal.
        - emails_with_attachments: Documents leaving via email attachments
          is the classic insider threat scenario.
        - avg_email_size: Large emails = large file transfers
        """
        if user_df.empty:
            return pd.Series({
                'email_count': 0,
                'external_email_count': 0,
                'emails_with_attachments': 0,
                'avg_email_size': 0,
            })
        
        # External = recipient doesn't have company domain
        # CERT uses dtaa.com as the company domain
        is_external = ~user_df['to'].str.contains(
            'dtaa.com', na=False
        ) if 'to' in user_df.columns else pd.Series([False]*len(user_df))
        
        return pd.Series({
            'email_count': len(user_df),
            'external_email_count': is_external.sum(),
            'emails_with_attachments': (user_df['attachments'] > 0).sum()
                                        if 'attachments' in user_df.columns else 0,
            'avg_email_size': user_df['size'].mean() 
                              if 'size' in user_df.columns else 0,
        })
    
    def _extract_http_features(self, user_df: pd.DataFrame) -> pd.Series:
        """
        WHY these HTTP features?
        
        Web browsing reveals intent. CERT scenarios include insiders who:
        - Visit job search sites (LinkedIn, Monster) before quitting
        - Visit competitor websites
        - Access cloud storage (Dropbox, Google Drive) to exfiltrate
        
        Simple URL category flags capture these patterns without needing
        a full URL classifier.
        """
        if user_df.empty:
            return pd.Series({
                'http_count': 0,
                'job_site_visits': 0,
                'cloud_storage_visits': 0,
            })
        
        urls = user_df['url'].str.lower() if 'url' in user_df.columns \
               else pd.Series([''] * len(user_df))
        
        job_keywords     = ['linkedin', 'monster', 'careerbuilder', 
                            'indeed', 'glassdoor', 'jobs']
        cloud_keywords   = ['dropbox', 'drive.google', 'onedrive', 
                            'box.com', 'wetransfer']
        
        return pd.Series({
            'http_count': len(user_df),
            'job_site_visits': urls.str.contains(
                '|'.join(job_keywords), na=False).sum(),
            'cloud_storage_visits': urls.str.contains(
                '|'.join(cloud_keywords), na=False).sum(),
        })
    
    def build_daily_features(self) -> pd.DataFrame:
        """
        For every (user, day) pair, compute the full feature vector
        by combining all 5 activity streams.
        
        WHY merge all streams? Because the threat signal is often
        multi-modal. A user might send one suspicious email AND
        plug in a USB AND access files after hours. No single stream
        captures that — but combined, it's a clear pattern.
        """
        if not hasattr(self, 'insiders'):
            raise RuntimeError('Call load_raw_data() before build_daily_features().')

        print("Building daily features for all users...")
        print("Aggregating logon activity...")
        features = self._aggregate_logon_features()

        aggregators = [
            ('device', self._aggregate_device_features),
            ('file', self._aggregate_file_features),
            ('email', self._aggregate_email_features),
            ('http', self._aggregate_http_features),
        ]
        for name, aggregate in aggregators:
            print(f'Aggregating {name} activity...')
            features = features.join(aggregate(), how='left')

        df = features.reset_index()
        feature_cols = [
            column for column in df.columns if column not in ['user', 'date']
        ]
        df[feature_cols] = df[feature_cols].fillna(0)
        if self.malicious_user_days:
            malicious_index = pd.MultiIndex.from_tuples(
                self.malicious_user_days, names=['user', 'date']
            )
            row_index = pd.MultiIndex.from_frame(df[['user', 'date']])
            df['is_malicious'] = row_index.isin(malicious_index).astype(np.int8)
        else:
            df['is_malicious'] = np.int8(0)

        cols = ['user', 'date', 'is_malicious'] + [
            column
            for column in df.columns
            if column not in ['user', 'date', 'is_malicious']
        ]
        df = df[cols].sort_values(['user', 'date']).reset_index(drop=True)
        
        print(f"Built feature matrix: {df.shape}")
        print(f"Feature columns: {[c for c in df.columns if c not in ['user','date','is_malicious']]}")
        print(f"Malicious days: {df['is_malicious'].sum()} / {len(df)}")
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, 
                           fit: bool = True) -> pd.DataFrame:
        """
        WHY normalize? Neural networks are sensitive to scale.
        A 'file_access_count' of 800 vs 'usb_connect_count' of 2
        would dominate the loss if not normalized.
        
        WHY StandardScaler specifically? It preserves the shape of
        the distribution and handles the heavy-tailed nature of
        behavioral data better than MinMax (which gets wrecked by outliers).
        
        WHY fit=True/False toggle? We fit the scaler on TRAINING data only.
        Fitting on test data would leak information about the test
        distribution into our normalization — a subtle but real data leak.
        """
        feature_cols = [c for c in df.columns 
                        if c not in ['user', 'date', 'is_malicious']]
        
        df_out = df.copy()
        if fit:
            df_out[feature_cols] = self.scaler.fit_transform(
                df[feature_cols].values
            )
        else:
            df_out[feature_cols] = self.scaler.transform(
                df[feature_cols].values
            )
        
        return df_out
    
    def save(self, df: pd.DataFrame, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / 'daily_features.csv', index=False)
        with (output_path / 'scaler.pkl').open('wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved to {output_path}")

class UserBehaviorDataset(Dataset):
    """
    Converts the daily feature matrix into overlapping sequences
    for LSTM training.
    
    WHY sequences? An LSTM needs temporal context. If we feed it
    one day at a time, it learns nothing about patterns across time.
    A 30-day window gives it enough context to understand "this user
    normally does X over a month" and flag deviations.
    
    WHY only BENIGN sequences for training? This is the core idea of
    anomaly detection via autoencoders:
    
    1. Train the model ONLY on normal behavior
    2. The model learns to reconstruct normal sequences well
    3. When it sees an anomalous sequence, reconstruction error is HIGH
    4. High reconstruction error = anomaly score
    
    We never show the model malicious data during training.
    If we did, it would learn to reconstruct malicious behavior too,
    and we'd lose the ability to detect it.
    """
    
    def __init__(
        self, 
        features_df: pd.DataFrame,
        window_size: int = 30,      # 30 days of context
        step_size: int = 1,          # slide window by 1 day at a time
        mode: str = 'train',         # 'train' uses only benign users
        benign_users: list | None = None,   # list of known-benign user IDs
    ):
        self.window_size = window_size
        self.mode = mode
        
        feature_cols = [c for c in features_df.columns 
                        if c not in ['user', 'date', 'is_malicious']]
        self.n_features = len(feature_cols)
        
        self.user_features = []
        self.window_user_indices = []
        self.window_starts = []
        self.labels = []   # 0 = benign window, 1 = contains malicious day
        self.users = []    # which user this sequence belongs to
        
        # Filter to only benign users for training
        if mode == 'train':
            users_to_use = benign_users if benign_users else \
                           features_df[features_df['is_malicious'] == 0]['user'].unique()
        else:
            users_to_use = features_df['user'].unique()
        
        print(f"Building {mode} sequences for {len(users_to_use)} users...")
        
        for user in users_to_use:
            user_data = features_df[features_df['user'] == user].sort_values('date')
            user_features = user_data[feature_cols].values.astype(np.float32)
            user_labels = user_data['is_malicious'].values
            
            # Need at least window_size days of data
            if len(user_features) < window_size:
                continue

            user_index = len(self.user_features)
            self.user_features.append(user_features)
            
            # Slide the window across this user's timeline
            for start in range(0, len(user_features) - window_size + 1, step_size):
                end = start + window_size
                label = user_labels[start:end].max()   # 1 if ANY day is malicious
                
                # In training mode, skip windows that contain malicious days
                if mode == 'train' and label == 1:
                    continue
                
                self.window_user_indices.append(user_index)
                self.window_starts.append(start)
                self.labels.append(label)
                self.users.append(user)
        
        self.window_user_indices = np.asarray(
            self.window_user_indices, dtype=np.int32
        )
        self.window_starts = np.asarray(self.window_starts, dtype=np.int32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

        if len(self.labels) == 0:
            raise ValueError(
                f'No {mode} sequences could be built. Reduce window_size or '
                'check the feature data.'
            )
        
        print(f"  Total sequences: {len(self.labels)}")
        print(f"  Malicious windows: {self.labels.sum()} "
              f"({self.labels.mean()*100:.1f}%)")
        print(f"  Sequence shape: ({self.window_size}, {self.n_features})")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        user_index = self.window_user_indices[idx]
        start = self.window_starts[idx]
        sequence = self.user_features[user_index][
            start:start + self.window_size
        ]
        return (
            torch.from_numpy(sequence),
            torch.LongTensor([self.labels[idx]])
        )
