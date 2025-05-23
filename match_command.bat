 C:\Users\ruby\chess\cutechess-1.3.1-win64\cutechess-cli.exe -engine name="Stockfish (1350)" cmd="C:\Users\ruby\chess\stockfish-windows-x86-64-avx2.exe" proto=uci option.UCI_LimitStrength=true option.UCI_Elo="350" st=0.1 timemargin=50 -engine name="Gemini Direct Use" cmd=C:\Users\ruby\projects\chessian\gemini_direct_engine.bat proto=uci st=60 stderr=gemini_err.log -pgnout match.pgn


 # 3^2 alpha beta with gemini vs. classic eval + moves
 C:\Users\ruby\chess\cutechess-1.3.1-win64\cutechess-cli.exe -engine name="classic alpha beta" cmd=C:\Users\ruby\projects\chessian\alpha_beta_engine.bat proto=uci st=120 -engine name="Gemini 2.0 Flash ab(3^2)" cmd=C:\Users\ruby\projects\chessian\alpha_beta_gem_engine.bat proto=uci st=120 stderr=gemini_err.log -rounds 2 -pgnout match_gem_20_flash_ab_vs_classic_ab.pgn -debug

 # 1000 nodes for each engine
 C:\Users\ruby\chess\cutechess-1.3.1-win64\cutechess-cli.exe -engine name="classic feature alpha beta" cmd=C:\Users\ruby\projects\chessian\ab_classic_features.bat proto=uci tc=inf nodes=1000 -engine name="Stockfish (1350)" cmd="C:\Users\ruby\chess\stockfish-windows-x86-64-avx2.exe" proto=uci option.UCI_LimitStrength=true option.UCI_Elo="1350" tc=inf nodes=1000 -rounds 10 -pgnout match_sf_classic_features4.pgn -debug