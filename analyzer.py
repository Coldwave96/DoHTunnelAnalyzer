import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

labels = []

print("[*] Loading datasets...")
# Load benign dataset
labels.append('benign')
benign_df = pd.read_csv("Datasets/Benign-DoH.csv").drop(labels=['Label'], axis=1)
benign_df['DoH'] = 0
benign_df = benign_df.rename(columns={'DoH': 'Label'})

# Load three types of DoH tunnel datasets & Concat datasets
labels.append('dns2tcp')
dns2tcp_df = pd.read_csv("Datasets/dns2tcp-DoH.csv").drop(labels=['DoH'], axis=1)
dns2tcp_df['labels'] = 1

labels.append('DNSCat2')
dnscat2_df = pd.read_csv("Datasets/DNSCat2-DoH.csv").drop(labels=['DoH'], axis=1)
dnscat2_df['labels'] = 2

labels.append('iodine')
iodine_df = pd.read_csv("Datasets/iodine-DoH.csv").drop(labels=['DoH'], axis=1)
iodine_df['labels'] = 3

malicious_df = pd.concat([dns2tcp_df, dnscat2_df, iodine_df], ignore_index=True)
malicious_df = malicious_df.rename(columns={'labels': 'Label'})

print("[*] Done!\n\n[*] Processing datasets...")
# Concat benign & malicious datasets
# Drop useless columns & data samples which contains Na or duplicates
mixed_df = shuffle(pd.concat([benign_df, malicious_df], ignore_index=True))
mixed_df = mixed_df.drop(labels=['SourceIP','DestinationIP','PacketTimeMode','TimeStamp'], axis=1)
mixed_df = mixed_df.dropna()
mixed_df = mixed_df.drop_duplicates()

# Create x & y
x = mixed_df.drop(labels=['Label'], axis=1)
y = mixed_df['Label'].values

# Standardizing datasets
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Stacked classifier design
def stack_classifier():
    base_model = []
    base_model.append(('RF', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)))
    base_model.append(('DT', DecisionTreeClassifier(random_state=42, max_depth=15)))

    meta_learner = MLPClassifier(hidden_layer_sizes=(100), activation='relu', max_iter=500, learning_rate='invscaling')

    model = StackingClassifier(estimators=base_model, final_estimator=meta_learner, cv=5)
    return model

# Split train/test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train
print("[*] Done!\n\n[*] Training...")
model = stack_classifier()
model = model.fit(x_train, y_train)

# Evaluation
print("[*] Done!\n\n[*] Evaluation")
y_pred = model.predict(x_test)
classification_report = classification_report(y_test, y_pred, target_names=labels)
print(classification_report)
