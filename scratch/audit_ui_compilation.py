import requests

vite_base = 'http://localhost:5173'

r_index = requests.get(f'{vite_base}/')
print('Vite Root Index Status:', r_index.status_code)

r_app = requests.get(f'{vite_base}/src/App.jsx')
print('Vite App.jsx Status:', r_app.status_code)

r_reg = requests.get(f'{vite_base}/src/pages/Register.jsx')
print('Vite Register.jsx Status:', r_reg.status_code)

r_ver = requests.get(f'{vite_base}/src/pages/Verify.jsx')
print('Vite Verify.jsx Status:', r_ver.status_code)

banned = ['YuNet', 'SFace', 'ArcFace', '512-D', 'Face Model', 'Candidate-only query']
ver_text = r_ver.text
reg_text = r_reg.text

print('\nAudit for banned terms in UI files:')
for term in banned:
    found_ver = term in ver_text
    found_reg = term in reg_text
    print(f'  Banned term "{term}": in Verify.jsx={found_ver}, in Register.jsx={found_reg}')

required_terms = [
    'IDENTITY VERIFICATION',
    'Identifying user...',
    'Biometric verification...',
    'Identity verified',
    'Identity verification failed'
]
print('\nAudit for required user-facing terminology in Verify.jsx:')
for term in required_terms:
    found = term in ver_text
    print(f'  Required term "{term}": found={found}')
