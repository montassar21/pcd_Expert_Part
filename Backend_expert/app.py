import os
from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_cors import CORS
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report ,roc_curve, auc,f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
from elasticsearch import Elasticsearch



es=Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme':'http'}])

app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"])
mapping = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "text": {"type": "text"},
            "user": {"type": "text"}
        }
    }
}
print(es.ping())
@app.route('/addComment',methods=['Post'])
def post():
    data=request.get_json()
    obj={
        'name':data['name'],
        'text':data['comment'],
        'user':'brahemmonta'
    }
    response={'Message':"Success !",'resp':obj}
    return jsonify(response)



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400
        
        # Get file and filename
        file = request.files['file']
        filename = secure_filename(file.filename)

        # Check if file format is supported
        file_format = os.path.splitext(filename)[1]
        if file_format not in ['.csv', '.txt', '.xlsx']:
            return jsonify({'error': 'Unsupported file format.'}), 400

        # Read file into pandas DataFrame
        df = file_formats[file_format](file)
        # Load the data and split it into train and test sets
        X = df.drop('Target', axis=1)
        y = df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
         #scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if file_format == '.xlsx':
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        testing_data_prediction=[]
        # train the logistic regression model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # make predictions on the test set
        y_pred = clf.predict(X_test)
        cf_matrix=confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        plt.savefig('../Frontend_expert/src/assets/LR_Cm.png')
        # predict probabilities for the test set
        y_pred_proba = clf.predict_proba(X_test)
        # calculate false positive rate, true positive rate and threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label="Dropout")
        # calculate AUC score
        auc_score = auc(fpr, tpr)
        # print(auc_score)
        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/LRCurve.png')
        plt.clf()
        # print('\nClassification Report:')
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap='Blues')
        plt.savefig('../Frontend_expert/src/assets/c_reportLR')
        plt.clf()
        # evaluate the model
        x_test_prediction=clf.predict(X_test)
        testing_data_prediction.append(accuracy_score(x_test_prediction,y_test))
        data=[accuracy_score(x_test_prediction,y_test),1-accuracy_score(x_test_prediction,y_test)]
        labels=['Accuracy','Error']
        colors = ['#54B6F3','gray']
        plt.title('Accuracy Score', fontsize=14)
        plt.legend(labels)
        plt.pie(data, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',    # here we also add the % sign 
    )
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/accuracyLR.png')
        plt.clf()


        # train the Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        # make predictions on the test set
        y_predRF = rf.predict(X_test)
        # print('Confusion Matrix:')
        rf_matrix=confusion_matrix(y_test, y_predRF)
        # print(rf_matrix)
        sns.heatmap(rf_matrix/np.sum(rf_matrix), annot=True,fmt='.2%', cmap='Reds')
        plt.savefig('../Frontend_expert/src/assets/RF_Cm.png')
        plt.clf()
        # print('\nClassification Report:')
        clf_report = classification_report(y_test, y_predRF, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap='Reds')
        plt.savefig('../Frontend_expert/src/assets/c_reportRF')
        plt.clf()
        # predict probabilities for the test set
        y_pred_probaRF = rf.predict_proba(X_test)

        # calculate false positive rate, true positive rate and threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probaRF[:, 1], pos_label="Dropout")
        # calculate AUC score
        auc_score = auc(fpr, tpr)
        # print(auc_score)
        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/RFCurve.png')
        plt.clf()
        x_test_prediction=rf.predict(X_test)
        testing_data_prediction.append(accuracy_score(x_test_prediction,y_test))
        data=[accuracy_score(x_test_prediction,y_test),1-accuracy_score(x_test_prediction,y_test)]
        labels=['Accuracy','Error']
        plt.title('Accuracy Score', fontsize=14)
        plt.legend(labels)
        colors = ['red','gray']
        plt.pie(data, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',    # here we also add the % sign 
    )
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/accuracyRF.png')
        plt.clf()
        # train the decision tree model
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        # make predictions on the test set
        y_predDT = dt.predict(X_test)
        # evaluate the model
        # print('Confusion Matrix:')
        dt_matrix = confusion_matrix(y_test, y_predDT)
        # print(dt_matrix)
        sns.heatmap(dt_matrix/np.sum(dt_matrix), annot=True,fmt='.2%', cmap='Greens')
        plt.savefig('../Frontend_expert/src/assets/DT_Cm.png')
        plt.clf()
        # print('\nClassification Report:')
        clf_report = classification_report(y_test, y_predDT, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap='Greens')
        plt.savefig('../Frontend_expert/src/assets/c_reportDT')
        plt.clf()
        # print('\nClassification Report:')
        # print(classification_report(y_test, y_pred))
        # predict probabilities for the test set
        y_pred_probaDT = dt.predict_proba(X_test)

        # calculate false positive rate, true positive rate and threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probaDT[:, 1], pos_label="Dropout")

        # calculate AUC score
        auc_score = auc(fpr, tpr)
        # print(auc_score)

        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/DTCurve.png')
        plt.clf()
        x_test_prediction=dt.predict(X_test)
        testing_data_prediction.append(accuracy_score(x_test_prediction,y_test))
        data=[accuracy_score(x_test_prediction,y_test),1-accuracy_score(x_test_prediction,y_test)]
        labels=['Accuracy','Error']
        plt.title('Accuracy Score', fontsize=14)
        colors = ['green','gray']
        plt.legend(labels)
        plt.pie(data, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',    # here we also add the % sign 
    )
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/accuracyDT.png')
        plt.clf()

        # train the Gradient boosting
        gbm = GradientBoostingClassifier(random_state=42)
        gbm.fit(X_train, y_train)
        # make predictions on the test set
        y_predGBM = gbm.predict(X_test)
        # evaluate the model
        # print('Confusion Matrix:')
        gbm_matrix = confusion_matrix(y_test, y_predGBM)
        # print(dt_matrix)
        sns.heatmap(gbm_matrix/np.sum(gbm_matrix), annot=True,fmt='.2%', cmap='Purples')
        plt.savefig('../Frontend_expert/src/assets/GBM_Cm.png')
        plt.clf()
        # print('\nClassification Report:')
        clf_report = classification_report(y_test, y_predGBM, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap="Purples")
        plt.savefig('../Frontend_expert/src/assets/c_reportGBM')
        plt.clf()
        # print('\nClassification Report:')
        # print(classification_report(y_test, y_pred))
        # predict probabilities for the test set
        y_pred_probaGBM = gbm.predict_proba(X_test)

        # calculate false positive rate, true positive rate and threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probaGBM[:, 1], pos_label="Dropout")
        # calculate AUC score
        auc_score = auc(fpr, tpr)
        # print(auc_score)
        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/GBMCurve.png')
        plt.clf()
        x_test_prediction=gbm.predict(X_test)
        testing_data_prediction.append(accuracy_score(x_test_prediction,y_test))
        data=[accuracy_score(x_test_prediction,y_test),1-accuracy_score(x_test_prediction,y_test)]
        labels=['Accuracy','Error']
        colors = ['#AAA1FF','grey']
        plt.title('Accuracy Score', fontsize=14)
        plt.legend(labels)
        plt.pie(data, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',    # here we also add the % sign 
    )
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/accuracyGBM.png')
        plt.clf()
        # Create SVM model object
        svm_model = svm.SVC(kernel='linear', probability=True)
        svm_model.fit(X_train, y_train)
        # make predictions on the test set
        y_predSVM = svm_model.predict(X_test)
        # evaluate the model
        # print('Confusion Matrix:')
        svm_matrix = confusion_matrix(y_test, y_predSVM)
        # print(svm_matrix)
        sns.heatmap(svm_matrix/np.sum(svm_matrix), annot=True,fmt='.2%', cmap='bone')
        # print('\nClassification Report:')
        # print(classification_report(y_test, y_pred))
        plt.savefig('../Frontend_expert/src/assets/SVM_Cm.png')
        plt.clf()
        # print('\nClassification Report:')
        clf_report = classification_report(y_test, y_predSVM, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap="bone")
        plt.savefig('../Frontend_expert/src/assets/c_reportSVM')
        plt.clf()
        # predict probabilities for the test set
        y_pred_probaSVM = svm_model.predict_proba(X_test)

        # calculate false positive rate, true positive rate and threshold values
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probaSVM[:, 1], pos_label="Dropout")

        # calculate AUC score
        auc_score = auc(fpr, tpr)
        # print(auc_score)
        # plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/SVMCurve.png')
        plt.clf()
        x_test_prediction=svm_model.predict(X_test)
        testing_data_prediction.append(accuracy_score(x_test_prediction,y_test))
        print(testing_data_prediction)
        data=[accuracy_score(x_test_prediction,y_test),1-accuracy_score(x_test_prediction,y_test)]
        labels=['Accuracy','Error']
        plt.title('Accuracy Score', fontsize=14)
        plt.legend(labels)
        colors = ['#97AFB5','gray']

        plt.pie(data, 
        labels=labels, 
        colors=colors,
        autopct='%.2f%%',    # here we also add the % sign 
    )
        # plt.show()
        plt.savefig('../Frontend_expert/src/assets/accuracySVM.png')
        plt.clf()
        # ...
        # Return success response
        comment=request.form.get('comment')
        return jsonify({'message': f'File {filename} uploaded successfully.', 'comment':comment,'accuracy':testing_data_prediction,'LR':['LR_Cm.png','LRCurve.png'],'RF':['RF_Cm.png','RFCurve.png'],'DT':['DT_Cm.png','DTCurve.png'],'GBM':['GBM_Cm.png','GBMCurve.png'],'SVM':['SVM_Cm.png','SVMCurve.png']}), 200

        # Convert DataFrame to CSV format
        csv_data = df.to_csv(index=False)

        # Return success response
        comment=request.form.get('comment')
        return jsonify({'message': f'File {filename} uploaded successfully.', 'comment': comment, 'data': csv_data}), 200
    
    except FileNotFoundError:
        return jsonify({'error': 'No file found.'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
