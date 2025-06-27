import pandas as pd

def enhanced_stock_selection(risk_profile, investment_amount):
    # Mock expanded dataset
    data = {
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Ent.', 'Zomato', 'Reliance', 'Bajaj Fin.', 'IRCTC'],
        'Sector': ['IT', 'Banking', 'IT', 'Infra', 'Tech', 'Energy', 'Finance', 'Travel'],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'P/E': [29, 21, 27, 42, 80, 31, 37, 65],
        'ROE': [24, 18, 22, 12, 3, 20, 21, 17],
        'Risk Category': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive',
                          'Moderate', 'Moderate', 'Aggressive']
    }
    df = pd.DataFrame(data)

    # Multi-factor scoring: you can tweak these weights
    df['Score'] = (
        (df['Sharpe Ratio'] / df['Beta']) * 0.4 +
        (1 / df['P/E']) * 0.2 +
        (df['ROE'] / 100) * 0.4
    )

    filtered = df[df['Risk Category'] == risk_profile].copy()
    filtered = filtered.sort_values(by='Score', ascending=False).head(4)

    filtered['Weight %'] = filtered['Score'] / filtered['Score'].sum() * 100
    filtered['Investment Amount (₹)'] = filtered['Weight %'] / 100 * investment_amount

    return filtered[['Stock', 'Sector', 'Sharpe Ratio', 'Beta', 'P/E', 'ROE', 'Weight %', 'Investment Amount (₹)']]
