# Stock Portfolio Optimization

FEUP AI project by 
- Anastasija Stojanoska
- Ahmet Caliskan
- Tommaso Cambursano



## Data

Data used for the project [here](https://www.kaggle.com/datasets/ashbellett/australian-historical-stock-prices)

## Run

To install all pip requirement:

```sh
pip install requirements.txt
```

To run, use:

```sh
streamlit run main.py
```


## Implemented Algorithm

- Hill-climbing
- Simulated Annealing
- Genetic Algorithm
- Tabu Serach


## Test created data

### ES1  
**Steady growth**, starting from **100** and reaching **130**.  

### ES2  
**Strong initial growth**, followed by a **sharp drop** halfway through the period.  

### ES3  
**Highly volatile**, with **random fluctuations**.  

### ES4  
A **market crash** occurs between **March 1, 2016**, and **March 5, 2016**.  

### ES5  
A **peak** is observed between **May 1, 2016**, and **May 5, 2016**.  

### ES6  
**Strong performance** in the **first half**, followed by a **decline** in the **second half**.  

### ES7  
**Weak performance** in the **first half**, followed by a **strong recovery** in the **second half**.  
