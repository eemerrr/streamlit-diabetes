import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# Order of features expected by the trained model (DIABETE4 is the target and DIABTYPE is unused)
FEATURE_ORDER = [
    "_RFHYPE6",
    "_RFCHOL3",
    "_CHOLCH3",
    "_BMI5",
    "SMOKE100",
    "CVDSTRK3",
    "_MICHD",
    "EXERANY2",
    "PRIMINS1",
    "MEDCOST1",
    "GENHLTH",
    "MENTHLTH",
    "PHYSHLTH",
    "DIFFWALK",
    "_SEX",
    "_AGEG5YR",
    "_EDUCAG",
    "_INCOMG1",
]

AGE_GROUP_OPTIONS: Dict[int, str] = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older",
}

EDUCATION_OPTIONS: Dict[int, str] = {
    1: "Did not graduate high school",
    2: "High school graduate",
    3: "Some college/technical school",
    4: "College graduate",
}

INCOME_OPTIONS: Dict[int, str] = {
    1: "<$15k",
    2: "$15k-<$25k",
    3: "$25k-<$35k",
    4: "$35k-<$50k",
    5: "$50k-<$75k",
    6: "$75k or more",
}

INSURANCE_OPTIONS: Dict[int, str] = {
    1: "Employer/union plan",
    2: "Self-purchased plan",
    3: "Medicare",
    4: "Medicaid",
    5: "TRICARE / VA / military",
    6: "Indian Health Service",
    7: "Other",
    8: "No insurance",
}


def _coerce_transform(preprocessor: Any, X: pd.DataFrame) -> Any:
    """Attempt to transform the features regardless of whether the object expects a DF or ndarray."""
    try:  # try the most permissive path first
        return preprocessor.transform(X)
    except Exception:
        return preprocessor.transform(X.values)


@st.cache_resource
def load_artifacts(
    model_path: Path = Path("best_model.pkl"),
    scaler_path: Path = Path("feature_scaler.pkl"),
) -> Tuple[object, str, float | None, Any | None]:
    """Load persisted estimator plus any attached preprocessor/scaler."""
    scaler: Any | None = None
    with model_path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        scaler = obj.get("preprocessor") or obj.get("scaler")
        model = obj.get("model", obj)
        model_name = obj.get("model_name", model.__class__.__name__)
        roc_auc = obj.get("roc_auc")
    else:
        model = obj
        model_name = obj.__class__.__name__
        roc_auc = None

    if scaler is None and scaler_path.exists():
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)

    return model, model_name, roc_auc, scaler


def yes_no(label: str, key: str, yes_value: int = 1, no_value: int = 0) -> int:
    """Binary input helper with yes/no semantics."""
    choice = st.radio(label, ["No", "Yes"], key=key, horizontal=True, index=0)
    return yes_value if choice == "Yes" else no_value


def build_input_row() -> Dict[str, float]:
    """Collect user inputs and map them into the feature vector order."""
    st.subheader("Health & lifestyle inputs")

    col1, col2 = st.columns(2)

    with col1:
        rfhype6 = yes_no("Hypertension diagnosis (_RFHYPE6)", "rfhype6")
        rfchol3 = yes_no("Told you have high cholesterol (_RFCHOL3)", "rfchol3")
        cholch3 = yes_no("Cholesterol checked in last 5 years (_CHOLCH3)", "cholch3")
        smoke100 = yes_no("Smoked >=100 cigarettes lifetime (SMOKE100)", "smoke100")
        cvdstrk3 = yes_no("History of stroke (CVDSTRK3)", "cvdstrk3")
        michd = yes_no("Coronary heart disease / heart attack (_MICHD)", "michd")
        exerany2 = yes_no("Any physical activity in last 30 days (EXERANY2)", "exerany2")
        medcost1 = yes_no("Skipped doctor due to cost (past 12 mo) (MEDCOST1)", "medcost1")
        diffwalk = yes_no("Difficulty walking/climbing stairs (DIFFWALK)", "diffwalk")
    with col2:
        sex_choice = st.radio("Sex (_SEX)", ["Female", "Male"], horizontal=True, key="sex")
        sex = 1 if sex_choice == "Male" else 0  # coded to 0/1 to match other binary indicators

        age_display = st.selectbox(
            "Age group (_AGEG5YR)",
            [f"{code}: {label}" for code, label in AGE_GROUP_OPTIONS.items()],
            index=4,
            key="age_group",
        )
        age_code = int(age_display.split(":")[0])

        prim_display = st.selectbox(
            "Primary health insurance (PRIMINS1)",
            [f"{code}: {label}" for code, label in INSURANCE_OPTIONS.items()],
            index=0,
            key="primins1",
        )
        primins1 = int(prim_display.split(":")[0])

        genhlth = st.select_slider(
            "General health (GENHLTH - 1=Excellent ... 5=Poor)",
            options=[1, 2, 3, 4, 5],
            value=3,
            key="genhlth",
        )

        menthlth = st.slider(
            "Days of poor mental health in past 30 days (MENTHLTH)",
            min_value=0,
            max_value=30,
            value=0,
            key="menthlth",
        )

        physhlth = st.slider(
            "Days of poor physical health in past 30 days (PHYSHLTH)",
            min_value=0,
            max_value=30,
            value=0,
            key="physhlth",
        )

        bmi = st.slider(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=60.0,
            value=27.0,
            step=0.1,
            key="bmi",
            help="Model expects BRFSS _BMI5 (BMI*10); this control accepts standard BMI.",
        )
        bmi5 = round(bmi * 10, 1)

        educ_display = st.selectbox(
            "Highest education (_EDUCAG)",
            [f"{code}: {label}" for code, label in EDUCATION_OPTIONS.items()],
            index=1,
            key="educag",
        )
        educag = int(educ_display.split(":")[0])

        income_display = st.selectbox(
            "Household income group (_INCOMG1)",
            [f"{code}: {label}" for code, label in INCOME_OPTIONS.items()],
            index=3,
            key="incomg1",
        )
        incomg1 = int(income_display.split(":")[0])

    return {
        "_RFHYPE6": rfhype6,
        "_RFCHOL3": rfchol3,
        "_CHOLCH3": cholch3,
        "_BMI5": bmi5,
        "SMOKE100": smoke100,
        "CVDSTRK3": cvdstrk3,
        "_MICHD": michd,
        "EXERANY2": exerany2,
        "PRIMINS1": primins1,
        "MEDCOST1": medcost1,
        "GENHLTH": genhlth,
        "MENTHLTH": menthlth,
        "PHYSHLTH": physhlth,
        "DIFFWALK": diffwalk,
        "_SEX": sex,
        "_AGEG5YR": age_code,
        "_EDUCAG": educag,
        "_INCOMG1": incomg1,
    }


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon=":bar_chart:", layout="wide")
    st.title("Diabetes Prediction (BRFSS-derived features)")
    st.caption(
        "Provide your details below. DIABETE4 (target) and DIABTYPE are excluded from inputs; "
        "all other variables are required by the trained model."
    )

    model, model_name, roc_auc, preprocessor = load_artifacts()
    st.success(f"Model loaded: {model_name}" + (f" (ROC-AUC: {roc_auc:.3f})" if roc_auc else ""))
    if preprocessor is None:
        st.warning(
            "No saved scaler/preprocessor detected. If the model was trained on standardized inputs, "
            "place the matching scaler in `feature_scaler.pkl` or embed it inside `best_model.pkl`."
        )

    with st.form("prediction_form"):
        inputs = build_input_row()
        submitted = st.form_submit_button("Predict diabetes status", type="primary")

    if submitted:
        ordered_values = [inputs[feat] for feat in FEATURE_ORDER]
        X = pd.DataFrame([ordered_values], columns=FEATURE_ORDER)
        model_input: Any = X
        if preprocessor is not None:
            try:
                model_input = _coerce_transform(preprocessor, X)
            except Exception as exc:  # pragma: no cover - defensive path for unexpected preprocessors
                st.error(f"Scaling failed: {exc}")
                st.stop()

        try:
            proba = float(model.predict_proba(model_input)[0][1])
            predicted_class = int(proba >= 0.5)
            st.subheader("Prediction")
            st.metric(
                "Estimated probability of diabetes",
                f"{proba:.1%}",
                help="Probability corresponds to DIABETE4=1 (doctor-diagnosed diabetes).",
            )
            if predicted_class == 1:
                st.error("Model predicts **diabetes** (positive). Please consult a clinician for medical guidance.")
            else:
                st.success("Model predicts **no diabetes** (negative). Maintain healthy habits and regular check-ups.")

            with st.expander("View encoded feature values sent to the model"):
                st.dataframe(X.T.rename(columns={0: "value"}))
        except Exception as exc:  # pragma: no cover - defensive UI path
            st.error(f"Prediction failed: {exc}")
            st.stop()


if __name__ == "__main__":
    main()
