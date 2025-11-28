# Business Context

"**Visit with Us**," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the **Wellness Tourism Package**, the company faces challenges in targeting the right customers efficiently.

The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an **MLOps pipeline**, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement.

This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.

---

# Objective

As an MLOps Engineer at **"Visit with Us,"** your responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases.

The primary objective is to build a model that predicts whether a customer will purchase the newly introduced **Wellness Tourism Package** before contacting them.

The pipeline will include:

- Data cleaning  
- Preprocessing  
- Transformation  
- Model building  
- Training  
- Evaluation  
- Deployment  

By leveraging **GitHub Actions** for CI/CD integration, the system will enable automated updates, streamline model deployment, and improve operational efficiency.

This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.

---

# Data Dictionary

The dataset contains customer and interaction data that serve as key attributes for predicting the likelihood of purchasing the Wellness Tourism Package.

---

## Customer Details

- **CustomerID:** Unique identifier for each customer.  
- **ProdTaken:** Target variable (0: No, 1: Yes).  
- **Age:** Age of the customer.  
- **TypeofContact:** Method of contact (Company Invited, Self Inquiry).  
- **CityTier:** City category (Tier 1 > Tier 2 > Tier 3).  
- **Occupation:** Customer's occupation.  
- **Gender:** Male/Female.  
- **NumberOfPersonVisiting:** Number of people accompanying the customer.  
- **PreferredPropertyStar:** Preferred hotel rating.  
- **MaritalStatus:** Single/Married/Divorced.  
- **NumberOfTrips:** Average annual trips.  
- **Passport:** Valid passport (0/1).  
- **OwnCar:** Owns a car (0/1).  
- **NumberOfChildrenVisiting:** Children below age 5.  
- **Designation:** Job designation.  
- **MonthlyIncome:** Gross monthly income.

---

## Customer Interaction Data

- **PitchSatisfactionScore:** Satisfaction score of sales pitch.  
- **ProductPitched:** Type of product pitched.  
- **NumberOfFollowups:** Total follow-ups post pitch.  
- **DurationOfPitch:** Duration of the pitch.

---

# Submission Guidelines

1. Submit a **single Python notebook in HTML (.html)** format.  
2. Plagiarism results in **zero marks**.  
3. Submit on time to avoid penalties.  
4. Read the problem statement and rubric carefully.  
5. Run code sequentially with observations.  
6. Ensure all outputs are visible.  
7. Add links to **Hugging Face Spaces** and **GitHub repo**.  
8. Non-executed code results in **50% deduction**.

---

# Rubric

## 1. Data Registration — 3 Points
- Create master folder with `data` subfolder  
- Register dataset on Hugging Face Dataset Space  

## 2. Data Preparation — 7 Points
- Load dataset from Hugging Face  
- Clean data  
- Split into train/test  
- Upload processed datasets back to Hugging Face  

## 3. Model Building with Experimentation Tracking — 13 Points
- Load data  
- Define model + parameters  
- Tune model  
- Log tuned parameters  
- Evaluate  
- Register best model  
- Allowed models: Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost  

## 4. Model Deployment — 11 Points
- Define Dockerfile  
- Load model from Hugging Face  
- Accept inputs & save as DataFrame  
- Define dependencies  
- Write hosting script for Hugging Face Space  

## 5. MLOps Pipeline with GitHub Actions — 15 Points
- Create `pipeline.yml`  
- List all ML steps  
- Push to GitHub  
- Automate workflow  
- Auto-update main branch  

## 6. Output Evaluation — 4 Points
- GitHub repo link  
- Screenshots of workflow  
- Hugging Face Space link  
- Streamlit app screenshot  

## 7. Notebook Overall Quality — 7 Points
- Structure, flow, comments  
- Visible outputs  
- No errors  
- Well-organized repo  
