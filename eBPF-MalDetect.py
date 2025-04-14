# -*- coding: utf-8 -*-

"""
eBPF 악성 프로그램 탐지 머신러닝 모델
instruction 패턴, helper 함수 호출, 제어 흐름 복잡도 등의 정적 피처를 기반으로
정상 및 악성 eBPF 프로그램을 분류하는 머신러닝 모델
"""

# 표준 라이브러리
import os
from datetime import datetime

# 서드파티 라이브러리
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# 한글 폰트 설정
def set_korean_font():
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False


# 한글 폰트 설정 함수 호출
set_korean_font()


def find_all_csv_files(directory):
    """
    디렉토리와 모든 하위 디렉토리에서 CSV 파일을 재귀적으로 찾습니다.

    Parameters:
    directory (str): 검색 시작 디렉토리 경로

    Returns:
    list: 발견된 모든 CSV 파일의 경로 리스트
    """
    csv_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    return csv_files


def load_datasets(normal_path, malicious_path):
    """
    정상 및 악성 eBPF 프로그램 데이터셋을 로드
    모든 하위 디렉토리에서 CSV 파일을 재귀적으로 검색

    CSV 파일은 다음 열을 포함해야 함:
    - filename: String
    - total_instructions: Integer
    - unique_opcodes: Integer
    - unique_mnemonics: Integer
    - memory_access_count: Integer
    - instruction_entropy: Float
    - conditional_jumps: Integer
    - unconditional_jumps: Integer
    - cyclomatic_complexity: Integer
    - function_count: Integer
    - function_names: String

    Parameters:
    normal_path (str): 정상 데이터셋 CSV 파일이 있는 디렉토리 경로
    malicious_path (str): 악성 데이터셋 CSV 파일이 있는 디렉토리 경로

    Returns:
    tuple: (X, y) 형태의 피처 매트릭스와 레이블 벡터
    """
    print("데이터셋 로드 중...")

    # 정상 데이터셋 로드 (모든 하위 디렉토리 포함)
    normal_files = find_all_csv_files(normal_path)
    print(f"발견된 정상 CSV 파일 수: {len(normal_files)}")
    normal_dfs = []

    for file in normal_files:
        try:
            df = pd.read_csv(file)
            df['label'] = 0  # 정상 데이터 레이블: 0
            normal_dfs.append(df)
            # print(f"로드됨: {file}")
        except Exception as e:
            print(f"경고: {file} 로드 중 오류 발생: {e}")

    if normal_dfs:
        normal_data = pd.concat(normal_dfs, ignore_index=True)
    else:
        raise ValueError(f"정상 데이터셋을 찾을 수 없습니다: {normal_path}")

    # 악성 데이터셋 로드 (모든 하위 디렉토리 포함)
    malicious_files = find_all_csv_files(malicious_path)
    print(f"발견된 악성 CSV 파일 수: {len(malicious_files)}")
    malicious_dfs = []

    for file in malicious_files:
        try:
            df = pd.read_csv(file)
            df['label'] = 1  # 악성 데이터 레이블: 1
            malicious_dfs.append(df)
            # print(f"로드됨: {file}")
        except Exception as e:
            print(f"경고: {file} 로드 중 오류 발생: {e}")

    if malicious_dfs:
        malicious_data = pd.concat(malicious_dfs, ignore_index=True)
    else:
        raise ValueError(f"악성 데이터셋을 찾을 수 없습니다: {malicious_path}")

    # 데이터셋 합치기
    all_data = pd.concat([normal_data, malicious_data], ignore_index=True)

    # 파일 정보 및 문자열 데이터 처리
    # filename은 제거하고, function_names는 토큰 수로 변환
    if 'function_names' in all_data.columns:
        all_data['function_name_count'] = all_data['function_names'].apply(lambda x: len(str(x).split(',')))

    # 피처와 레이블 분리 <정적 피처 4가지만 사용>
    feature_cols = ['total_instructions', 'cyclomatic_complexity', 'conditional_jumps', 'instruction_entropy']
    # feature_cols = ['total_instructions', 'cyclomatic_complexity', 'conditional_jumps']
    # feature_cols = ['total_instructions', 'cyclomatic_complexity', 'instruction_entropy']
    # feature_cols = ['total_instructions', 'conditional_jumps', 'instruction_entropy']
    # feature_cols = ['cyclomatic_complexity', 'conditional_jumps', 'instruction_entropy']

    # 존재하는 열만 사용
    available_features = [col for col in feature_cols if col in all_data.columns]

    X = all_data[available_features]
    y = all_data['label']

    # 데이터셋 정보 출력
    print(f"데이터셋 로드 완료: 총 {len(all_data)} 샘플, 정상: {len(normal_data)}, 악성: {len(malicious_data)}")
    print(f"사용된 피처 ({len(available_features)}개): {', '.join(available_features)}")

    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42, class_ratio=0.8):
    """
    가능한 많은 데이터를 유지하며 정상:악성 비율을 먼저 8:2로 맞춘 후,
    그 balanced 데이터를 기반으로 train/test를 stratified split

    Parameters:
        X (DataFrame): 피처 데이터
        y (Series): 레이블
        test_size (float): 테스트셋 비율
        random_state (int): 시드
        class_ratio (float): 정상 비율 (예: 0.8 → 8:2)

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    print(f"정상:악성 = {class_ratio}:{1 - class_ratio} 비율로 최대한 많이 살려서 조정 중...")

    df = X.copy()
    df['label'] = y

    df_normal = df[df['label'] == 0]
    df_malicious = df[df['label'] == 1]

    total_normal = len(df_normal)
    total_malicious = len(df_malicious)

    # 1. 가능한 최대 비율에 맞는 정수 개수 계산
    k_normal = int(total_normal / class_ratio)
    k_malicious = int(total_malicious / (1 - class_ratio))
    k = min(k_normal, k_malicious)

    # 2. 전체를 10의 배수로 맞춰 딱 나눠떨어지게 조절
    total_target = int((k * class_ratio + k * (1 - class_ratio)) // 10 * 10)
    used_normal = int(total_target * class_ratio)
    used_malicious = total_target - used_normal

    df_normal_sampled = df_normal.sample(n=used_normal, random_state=random_state)
    df_malicious_sampled = df_malicious.sample(n=used_malicious, random_state=random_state)

    # 합치기
    df_balanced = pd.concat([df_normal_sampled, df_malicious_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"균형 조정 완료 → 정상 {len(df_normal_sampled)}개, 악성 {len(df_malicious_sampled)}개")
    print(f"총 샘플 수: {len(df_balanced)} / 클래스 분포: {df_balanced['label'].value_counts().to_dict()}")

    # 훈련/테스트 분할 (stratified)
    X_balanced = df_balanced.drop(columns=['label'])
    y_balanced = df_balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, stratify=y_balanced, random_state=random_state
    )

    print(f"훈련 데이터 크기: {len(X_train)} / 클래스 분포: {y_train.value_counts().to_dict()}")
    print(f"테스트 데이터 크기: {len(X_test)} / 클래스 분포: {y_test.value_counts().to_dict()}")

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    여러 머신러닝 모델 훈련 및 평가 (Threshold 0.5, 안정된 구조)
    """
    print("모델 훈련 및 평가 중...")

    threshold = 0.5  # 기본 threshold

    # 모델 정의
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(),
        'NeuralNetwork': MLPClassifier(random_state=42, early_stopping=True, max_iter=1000)
    }

    # 하이퍼파라미터 그리드 (NeuralNetwork 구조 간소화됨)
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2']
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'criterion': ['gini', 'entropy']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'NeuralNetwork': {
            'hidden_layer_sizes': [(5,), (10,), (5, 5)],
            'activation': ['relu'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'early_stopping': [False],
            'max_iter': [2000]
        }       
    }

    results = {}
    best_auc = 0
    best_model_name = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{name} 모델 훈련 중...")

        grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='f1', n_jobs=-1, verbose=1)

        try:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        except Exception as e:
            print(f"경고: {name} 모델 훈련 중 오류 발생: {e}")
            best_model = model
            best_params = "기본값"

        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
            print(f"{name} 교차검증 F1-score 평균: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)

            print(f"{name} 최적 파라미터: {best_params}")
            print(f"{name} 정확도: {accuracy:.4f}, F1-score: {report['1']['f1-score']:.4f}, AUC: {roc_auc:.4f}")

            results[name] = {
                'model': best_model,
                'params': best_params,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'roc_curve': (fpr, tpr, roc_auc),
                'pr_curve': (precision, recall)
            }

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model_name = name
        except Exception as e:
            print(f"경고: {name} 모델 평가 중 오류 발생: {e}")

    if best_model_name:
        print(f"\n최고 성능 모델: {best_model_name}, AUC: {best_auc:.4f}")
    else:
        print("\n경고: 모든 모델 훈련에 실패하였습니다.")
        for name in models.keys():
            if name in results:
                best_model_name = name
                break

    return results, best_model_name



def visualize_results(results, X, feature_names=None, output_dir='results'):
    """
    분석 결과 시각화 및 저장

    Parameters:
    results (dict): 모델 훈련 및 평가 결과
    X (DataFrame): 원본 피처 데이터
    feature_names (list, optional): 피처 이름 리스트
    output_dir (str): 결과 저장 디렉토리
    """
    print("결과 시각화 중...")

    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 현재 시간을 문자열로 변환하여 파일명에 사용
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모델별 결과 시각화
    for name, result in results.items():
        plt.figure(figsize=(16, 12))

        # 1. 혼동 행렬
        plt.subplot(2, 2, 1)
        conf_matrix = result['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['정상', '악성'], yticklabels=['정상', '악성'])
        plt.title(f'{name} 혼동 행렬')
        plt.ylabel('실제')
        plt.xlabel('예측')

        # 2. ROC 커브
        plt.subplot(2, 2, 2)
        fpr, tpr, roc_auc = result['roc_curve']
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC 커브')
        plt.legend(loc='lower right')

        # 3. Precision-Recall 커브
        plt.subplot(2, 2, 3)
        precision, recall = result['pr_curve']
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} Precision-Recall 커브')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_results_{timestamp}.png'), dpi=300)
        plt.close()

    # 모델 비교 그래프
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    f1_scores = [results[name]['classification_report']['1']['f1-score'] for name in model_names]
    accuracies = [results[name]['accuracy'] for name in model_names]
    auc_scores = [results[name]['roc_curve'][2] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.bar(x - width, accuracies, width, label='정확도')
    plt.bar(x, f1_scores, width, label='F1 점수')
    plt.bar(x + width, auc_scores, width, label='AUC')

    plt.xlabel('모델')
    plt.ylabel('점수')
    plt.title('모델 성능 비교')
    plt.xticks(x, model_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_{timestamp}.png'), dpi=300)
    plt.close()

    print(f"시각화 결과 저장 완료: {output_dir}")


def save_models(results, scaler, selector, output_dir='models'):
    """
    훈련된 모델, 스케일러, 특성 선택기 저장

    Parameters:
    results (dict): 모델 훈련 및 평가 결과
    scaler (StandardScaler): 피처 스케일러
    selector (SelectKBest): 특성 선택기
    output_dir (str): 모델 저장 디렉토리
    """
    print("모델 저장 중...")

    # 모델 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 현재 시간을 문자열로 변환하여 파일명에 사용
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모델 저장
    for name, result in results.items():
        model_path = os.path.join(output_dir, f'{name}_model_{timestamp}.joblib')
        joblib.dump(result['model'], model_path)
        print(f"{name} 모델 저장 완료: {model_path}")

    # 스케일러 저장
    scaler_path = os.path.join(output_dir, f'scaler_{timestamp}.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"스케일러 저장 완료: {scaler_path}")

    # 특성 선택기 저장
    if selector is not None:
        selector_path = os.path.join(output_dir, f'selector_{timestamp}.joblib')
        joblib.dump(selector, selector_path)
        print(f"특성 선택기 저장 완료: {selector_path}")


def save_datasets_to_csv(X_train, X_test, y_train, y_test, output_dir):
    """
    전처리된 훈련 및 테스트 데이터셋을 CSV로 저장

    Parameters:
    X_train (DataFrame): 훈련 데이터 피처
    X_test (DataFrame): 테스트 데이터 피처
    y_train (Series): 훈련 데이터 레이블
    y_test (Series): 테스트 데이터 레이블
    output_dir (str): 출력 디렉토리 경로

    Returns:
    str: 데이터셋이 저장된 디렉토리 경로
    """
    print("데이터셋 CSV 저장 중...")

    # 데이터셋 저장 디렉토리 생성
    dataset_dir = os.path.join(output_dir, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    # 현재 시간을 문자열로 변환하여 파일명에 사용
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 훈련 데이터 저장
    X_train_with_labels = X_train.copy()
    X_train_with_labels['label'] = y_train
    train_path = os.path.join(dataset_dir, f'train_data_{timestamp}.csv')
    X_train_with_labels.to_csv(train_path, index=False)

    # 테스트 데이터 저장
    X_test_with_labels = X_test.copy()
    X_test_with_labels['label'] = y_test
    test_path = os.path.join(dataset_dir, f'test_data_{timestamp}.csv')
    X_test_with_labels.to_csv(test_path, index=False)

    print(f"훈련 데이터 저장됨: {train_path}")
    print(f"테스트 데이터 저장됨: {test_path}")

    return dataset_dir


def main():
    """
    메인 함수: eBPF 정적 분석 기반 악성 탐지 머신러닝 워크플로우를 실행합니다.
    - 데이터 로드, 전처리, 학습, 평가, 시각화, 모델 저장
    """

    # ==== 사용자 설정 변수 ====
    normal_path = '/root/ml/Obj_dataset/dataset/nomal'            # 정상 데이터 디렉토리 경로
    malicious_path = '/root/ml/Obj_dataset/dataset/malicious'      # 악성 데이터 디렉토리 경로
    output_dir = '/root/ml/output'                  # 결과 저장 경로
    test_size = 0.3                          # 테스트셋 비율
    seed = 42                                # 랜덤 시드
    only_confusion_matrix = False            # True면 혼동 행렬만 출력
    # ==========================

    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. 데이터셋 로드
    print("\n===== 1. 데이터셋 로드 =====")
    X, y = load_datasets(normal_path, malicious_path)

    print("\n데이터셋 통계:")
    print(X.describe())

    # 2. 데이터 전처리
    print("\n===== 2. 데이터 전처리 =====")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=test_size, random_state=seed, class_ratio=0.7)


    # 2.1 훈련 및 테스트 데이터셋을 CSV로 저장
    dataset_dir = save_datasets_to_csv(pd.DataFrame(X_train, columns=X.columns),
                                       pd.DataFrame(X_test, columns=X.columns),
                                       y_train, y_test, output_dir)

    # 3. 사용된 피처 출력
    print("\n===== 3. 사용된 피처 =====")
    X_train_selected = X_train
    X_test_selected = X_test
    selector = None
    selected_features = X.columns.tolist()
    print(f"사용된 피처 ({len(selected_features)}개): {', '.join(selected_features)}")


    # 4. 모델 훈련 및 평가
    print("\n===== 4. 모델 훈련 및 평가 =====")
    results, best_model_name = train_and_evaluate_models(X_train_selected, X_test_selected, y_train, y_test)

    # 5. 결과 시각화
    if not only_confusion_matrix:
        print("\n===== 5. 결과 시각화 =====")
        visualize_results(results, X, feature_names=X.columns.tolist(), output_dir=results_dir)

    # 6. 모델 저장
    print("\n===== 6. 모델 저장 =====")
    save_models(results, scaler, selector, output_dir=models_dir)

    # 7. 요약 결과 출력
    print("\n===== 결과 요약 =====")
    print(f"총 데이터 샘플 수: {len(X)}")
    print(f"사용된 피처: {', '.join(X.columns)}")

    # 각 모델의 혼동 행렬 출력
    for model_name, result in results.items():
        print(f"\n===== {model_name} 평가 결과 =====")
        conf_matrix = result['confusion_matrix']
        tn, fp, fn, tp = conf_matrix.ravel()

        accuracy = result['accuracy']
        f1 = result['classification_report']['1']['f1-score']
        auc_score = result['roc_curve'][2]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"정확도(Accuracy): {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"정밀도(Precision): {precision:.4f}")
        print(f"재현율(Recall): {recall:.4f}")
        print(f"혼동 행렬:\n{conf_matrix}")


    # 8. 모델 해석 (옵션)
    print(f"\n최고 성능 모델: {best_model_name}")
    print(f"정확도: {results[best_model_name]['accuracy']:.4f}")
    print(f"F1-score: {results[best_model_name]['classification_report']['1']['f1-score']:.4f}")
    print(f"AUC: {results[best_model_name]['roc_curve'][2]:.4f}")
    print(f"\n결과 저장 경로: {output_dir}")
    print(f"데이터셋 저장 경로: {dataset_dir}")
    print("프로그램 실행 완료!")


if __name__ == "__main__":
    main()
