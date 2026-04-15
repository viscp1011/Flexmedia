@echo off
echo ============================================
echo   FLEXMEDIA - Sistema de Analise de Totem
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Instalando dependencias...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERRO ao instalar dependencias. Verifique se o Python esta instalado.
    pause
    exit /b 1
)
echo Dependencias OK!
echo.

echo [2/3] Gerando dados simulados (caso nao exista conexao Oracle)...
python gera_dados.py
echo.

echo [3/3] Iniciando dashboard...
echo Acesse: http://localhost:8501
echo Para encerrar pressione CTRL+C
echo.
python -m streamlit run dashboard_flexmedia.py

pause
