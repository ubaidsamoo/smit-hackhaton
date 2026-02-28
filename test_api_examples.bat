@echo off
REM Flight Delay Prediction API - cURL Examples
REM Run these commands to test the API endpoints

setlocal enabledelayedexpansion
set BASE_URL=http://127.0.0.1:8000

echo.
echo ================================================================================
echo             FLIGHT DELAY PREDICTION API - CURL EXAMPLES
echo ================================================================================
echo.
echo Make sure the API is running before executing these commands:
echo    python flight_delay_api.py
echo.

:MENU
echo.
echo ================================================================================
echo Select a test to run:
echo ================================================================================
echo 1. Health Check
echo 2. Model Information
echo 3. Feature Importance
echo 4. Single Prediction (Low Risk)
echo 5. Single Prediction (High Risk)
echo 6. Batch Predictions
echo 7. Prediction Explanation
echo 8. Run All Tests
echo 0. Exit
echo.
set /p choice="Enter your choice (0-8): "

if "%choice%"=="0" goto END
if "%choice%"=="1" goto TEST1
if "%choice%"=="2" goto TEST2
if "%choice%"=="3" goto TEST3
if "%choice%"=="4" goto TEST4
if "%choice%"=="5" goto TEST5
if "%choice%"=="6" goto TEST6
if "%choice%"=="7" goto TEST7
if "%choice%"=="8" goto TEST_ALL

echo Invalid choice. Please try again.
goto MENU

:TEST1
cls
echo ================================================================================
echo TEST 1: HEALTH CHECK
echo ================================================================================
echo.
curl -X GET "%BASE_URL%/health"
echo.
echo.
pause
goto MENU

:TEST2
cls
echo ================================================================================
echo TEST 2: MODEL INFORMATION
echo ================================================================================
echo.
curl -X GET "%BASE_URL%/model-info"
echo.
echo.
pause
goto MENU

:TEST3
cls
echo ================================================================================
echo TEST 3: FEATURE IMPORTANCE
echo ================================================================================
echo.
curl -X GET "%BASE_URL%/feature-importance"
echo.
echo.
pause
goto MENU

:TEST4
cls
echo ================================================================================
echo TEST 4: SINGLE PREDICTION (LOW RISK FLIGHT)
echo ================================================================================
echo.
echo Request:
echo.
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"day_of_month\": 10, \"day_of_week\": 2, \"op_carrier_airline_id\": 19977, \"dest_airport_id\": 12892, \"distance\": 600.0, \"dep_delay\": 2.0, \"taxi_out\": 12.0, \"taxi_in\": 6.0}"
echo.
echo.
pause
goto MENU

:TEST5
cls
echo ================================================================================
echo TEST 5: SINGLE PREDICTION (HIGH RISK FLIGHT)
echo ================================================================================
echo.
echo Request:
echo.
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"day_of_month\": 20, \"day_of_week\": 5, \"op_carrier_airline_id\": 20363, \"dest_airport_id\": 11433, \"distance\": 1200.0, \"dep_delay\": 45.0, \"taxi_out\": 25.0, \"taxi_in\": 12.0}"
echo.
echo.
pause
goto MENU

:TEST6
cls
echo ================================================================================
echo TEST 6: BATCH PREDICTIONS
echo ================================================================================
echo.
curl -X POST "%BASE_URL%/predict-batch" ^
  -H "Content-Type: application/json" ^
  -d "{\"flights\": [{\"day_of_month\": 5, \"day_of_week\": 1, \"op_carrier_airline_id\": 19977, \"dest_airport_id\": 12892, \"distance\": 500.0, \"dep_delay\": 3.0, \"taxi_out\": 13.0, \"taxi_in\": 7.0}, {\"day_of_month\": 15, \"day_of_week\": 3, \"op_carrier_airline_id\": 20363, \"dest_airport_id\": 11433, \"distance\": 1000.0, \"dep_delay\": 25.0, \"taxi_out\": 18.0, \"taxi_in\": 9.0}, {\"day_of_month\": 25, \"day_of_week\": 6, \"op_carrier_airline_id\": 20366, \"dest_airport_id\": 10397, \"distance\": 800.0, \"dep_delay\": 50.0, \"taxi_out\": 28.0, \"taxi_in\": 11.0}]}"
echo.
echo.
pause
goto MENU

:TEST7
cls
echo ================================================================================
echo TEST 7: PREDICTION EXPLANATION
echo ================================================================================
echo.
curl -X POST "%BASE_URL%/explain" ^
  -H "Content-Type: application/json" ^
  -d "{\"day_of_month\": 22, \"day_of_week\": 4, \"op_carrier_airline_id\": 19977, \"dest_airport_id\": 12892, \"distance\": 1500.0, \"dep_delay\": 55.0, \"taxi_out\": 30.0, \"taxi_in\": 12.0}"
echo.
echo.
pause
goto MENU

:TEST_ALL
cls
echo ================================================================================
echo Running all tests...
echo ================================================================================
echo.

echo TEST 1: HEALTH CHECK
curl -X GET "%BASE_URL%/health"
echo.
echo.

timeout /t 2 /nobreak

echo TEST 2: MODEL INFORMATION
curl -X GET "%BASE_URL%/model-info"
echo.
echo.

timeout /t 2 /nobreak

echo TEST 3: FEATURE IMPORTANCE (Top 5 only)
curl -X GET "%BASE_URL%/feature-importance"
echo.
echo.

timeout /t 2 /nobreak

echo TEST 4: SINGLE PREDICTION (Low Risk)
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"day_of_month\": 10, \"day_of_week\": 2, \"op_carrier_airline_id\": 19977, \"dest_airport_id\": 12892, \"distance\": 600.0, \"dep_delay\": 2.0, \"taxi_out\": 12.0, \"taxi_in\": 6.0}"
echo.
echo.

timeout /t 2 /nobreak

echo TEST 5: SINGLE PREDICTION (High Risk)
curl -X POST "%BASE_URL%/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"day_of_month\": 20, \"day_of_week\": 5, \"op_carrier_airline_id\": 20363, \"dest_airport_id\": 11433, \"distance\": 1200.0, \"dep_delay\": 45.0, \"taxi_out\": 25.0, \"taxi_in\": 12.0}"
echo.
echo.

pause
goto MENU

:END
echo.
echo ================================================================================
echo Thank you for using Flight Delay Prediction API!
echo ================================================================================
echo.
endlocal
