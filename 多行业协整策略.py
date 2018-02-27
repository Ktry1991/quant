# 作者：heying
# 策略基于joinquant平台实现
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from six import StringIO

#===========================================

def initialize(context):
    #获取申万3级行业code
    #g.SW3 = ['801780']
    #g.SW3 = ['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150','801160','801170','801180','801200','801210','801230','801710','801720','801730','801740','801750','801760','801770','801780','801790','801880','801890']#['801780', '801170','801160','801150']#['801011','801012','801013','801014','801015','801016','801017','801018','801021','801022','801023','801024','801032','801033','801034','801035','801036','801037','801041','801051','801053','801054','801055','801072','801073','801074','801075','801076','801081','801082','801083','801084','801085','801092','801093','801094','801101','801102','801111','801112','801123','801131','801132','801141','801142','801143','801144','801151','801152','801153','801154','801155','801156','801161','801162','801163','801164','801171','801172','801173','801174','801175','801176','801177','801178','801181','801182','801191','801192','801193','801194','801202','801203','801204','801205','801211','801212','801213','801214','801215','801222','801223','801231','801711','801712','801713','801721','801722','801723','801724','801725','801731','801732','801733','801734','801741','801742','801743','801744','801751','801752','801761','801881']#['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150','801160','801170','801180','801200','801210','801230','801710','801720','801730','801740','801750','801760','801770','801780','801790','801880','801890']#['HY493']#['852121']#,'801780','851941'] #['850111','850112','850113','850121','850122','850131','850141','850151','850152','850154','850161','850171','850181','850211','850221','850222','850231','850241','850242','850311','850313','850321','850322','850323','850324','850331','850332','850333','850334','850335','850336','850337','850338','850339','850341','850342','850343','850344','850345','850351','850352','850353','850361','850362','850363','850372','850373','850381','850382','850383','850411','850412','850521','850522','850523','850531','850541','850542','850543','850544','850551','850552','850553','850611','850612','850614','850615','850616','850623','850711','850712','850713','850714','850715','850716','850721','850722','850723','850724','850725','850726','850727','850728','850729','850731','850741','850751','850811','850812','850813','850822','850823','850831','850832','850833','850841','850851','850852','850911','850912','850913','850921','850935','850936','850941','851012','851013','851014','851021','851111','851112','851113','851114','851115','851121','851122','851231','851232','851233','851234','851235','851236','851241','851242','851243','851244','851311','851312','851313','851314','851315','851316','851322','851323','851324','851325','851326','851327','851411','851421','851432','851433','851434','851435','851441','851511','851512','851521','851531','851541','851551','851561','851611','851612','851613','851614','851615','851621','851631','851641','851711','851721','851731','851741','851751','851761','851771','851781','851811','851821','851911','851921','851931','851941','852021','852031','852032','852033','852041','852051','852052','852111','852112','852121','852131','852141','852151','852211','852221','852222','852223','852224','852225','852226','852241','852242','852243','852244','852311','857221','857231','857232','857233','857234','857235','857241','857242','857243','857244','857251','857321','857322','857323','857331','857332','857333','857334','857335','857336','857341','857342','857343','857344','857411','857421','857431','858811']
    #['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150','801160','801170','801180','801200','801210','801230','801710','801720','801730','801740','801750','801760','801770','801780','801790','801880','801890']#['HY493']#['852121']#,'801780','851941'] #['850111','850112','850113','850121','850122','850131','850141','850151','850152','850154','850161','850171','850181','850211','850221','850222','850231','850241','850242','850311','850313','850321','850322','850323','850324','850331','850332','850333','850334','850335','850336','850337','850338','850339','850341','850342','850343','850344','850345','850351','850352','850353','850361','850362','850363','850372','850373','850381','850382','850383','850411','850412','850521','850522','850523','850531','850541','850542','850543','850544','850551','850552','850553','850611','850612','850614','850615','850616','850623','850711','850712','850713','850714','850715','850716','850721','850722','850723','850724','850725','850726','850727','850728','850729','850731','850741','850751','850811','850812','850813','850822','850823','850831','850832','850833','850841','850851','850852','850911','850912','850913','850921','850935','850936','850941','851012','851013','851014','851021','851111','851112','851113','851114','851115','851121','851122','851231','851232','851233','851234','851235','851236','851241','851242','851243','851244','851311','851312','851313','851314','851315','851316','851322','851323','851324','851325','851326','851327','851411','851421','851432','851433','851434','851435','851441','851511','851512','851521','851531','851541','851551','851561','851611','851612','851613','851614','851615','851621','851631','851641','851711','851721','851731','851741','851751','851761','851771','851781','851811','851821','851911','851921','851931','851941','852021','852031','852032','852033','852041','852051','852052','852111','852112','852121','852131','852141','852151','852211','852221','852222','852223','852224','852225','852226','852241','852242','852243','852244','852311','857221','857231','857232','857233','857234','857235','857241','857242','857243','857244','857251','857321','857322','857323','857331','857332','857333','857334','857335','857336','857341','857342','857343','857344','857411','857421','857431','858811']
    g.days = 0 #记录当前交易天数
    set_params()
    set_variables()
    set_backtest()


# ---代码块1. 设置参数
def set_params():
    # 基准
    g.benchmark = '300618.XSHE'
    # 调仓周期
    g.trade_freq = 3
    # 每个行业的候选股票数量
    g.stock_num_ind = 8
    # 股票1默认仓位
    g.p = 0.5
    # 股票2默认仓位
    g.q = 0.5
    # 算z-score天数
    g.test_days = 7
    # 计算p值的历史天数窗口
    g.p_value_days = 7
    # 计算coef的历史天数窗口
    g.coef_days = 50
    # z_score 调仓上界
    g.z_score_up_bound = 0.5
    # z_score 调仓下界
    g.z_score_low_bound = -0.5
    # 是否开启止损方法
    g.open_stop_loss = True
    # 止损天数
    g.stop_days = 0
    # 股票池1
    g.stock_list1 = []#['601318.XSHG','000001.XSHE','601288.XSHG','601633.XSHG','600104.XSHG','000538.XSHE','601688.XSHG','002460.XSHE']
    # 股票池2
    g.stock_list2 = []#['601336.XSHG','002142.XSHE','601988.XSHG','000625.XSHE','601238.XSHG','002424.XSHE','600999.XSHG','002466.XSHE']
    # 股票池3
    g.stock_list2 = []
# ---代码块2. 设置变量
def set_variables():
    # 股票1
    g.security1 = '600720.XSHG'#'600192.XSHG'    
    # 股票2
    g.security2 = '600449.XSHG'#'600088.XSHG'
    # 回归系数
    g.regression_ratio = 1.0036
    # 现在状态
    g.state = 'empty'


# ---代码块3. 设置回测
def set_backtest():
    # 设置基准
    set_benchmark(g.benchmark)
    # 只报错
    log.set_level('order', 'error')
    # 真实价格
    set_option('use_real_price', True) 
    # 无滑点
    set_slippage(FixedSlippage(0.))


# ---代码块4.计算z-score
def z_test(context):
    # 获取两支股票历史价格
    prices1 = np.array(attribute_history(g.security1, g.test_days, '1d', ['close']).close)
    prices2 = np.array(attribute_history(g.security2, g.test_days, '1d', ['close']).close)
    # 根据回归比例算它们的平稳序列 Y-a.X
    g.regression_ratio = get_coef_by_ols(context,[g.security1, g.security2])
    stable_series = prices2 - g.regression_ratio*prices1
    # 算均值
    series_mean = mean(stable_series)
    # 算标准差
    sigma = np.std(stable_series)
    # 算序列现值离均值差距多少
    diff = stable_series[-1] - series_mean
    # 返回z值
    return(diff/sigma)

# ---代码块5.获取信号
# 返回新的状态，是一个string
def get_signal(context):
    z_score = z_test(context)
    if z_score > g.z_score_up_bound:
        # 状态为全仓第一支
        return('buy1')
    # 如果小于负标准差
    if z_score < g.z_score_low_bound:
        # 状态为全仓第二支
        return('buy2')
    # 如果在正负标准差之间
    if g.z_score_low_bound <= z_score <= g.z_score_up_bound:
        # 如果差大于0
        if z_score >= 0:
            # 在均值上面
            return('side1')
        # 反之
        else:
            # 在均值下面
            return('side2')

# ---代码块6.根据信号调换仓位
# 输入是目标状态，输入为一个string
def change_positions(new_state,context):
    for s in context.portfolio.positions:
        if s not in g.stock_list1 and s not in g.stock_list2:
            order_target(s, 0)
    # 总值产价值
    total_value = context.portfolio.portfolio_value*(1.0/3)
    # print(total_value, (1.0/len(g.stock_list1)))
    # 如果新状态是全仓股票1
    if new_state == 'buy1':
        # 全卖股票2
        order_target(g.security2, 0)
        # 全买股票1
        order_target_value(g.security1, total_value)
        # 旧状态更改
        g.state = 'buy1'
    # 如果新状态是全仓股票2
    if new_state == 'buy2':
        # 全卖股票1
        order_target(g.security1, 0)
        # 全买股票2
        order_target_value(g.security2, total_value)
        # 旧状态更改
        g.state = 'buy2'
    # 如果处于全仓一股票状态，但是z-score交叉0点
    if (g.state == 'buy1' and new_state == 'side2') or (g.state == 'buy2' and new_state == 'side1'):
        # 按照p,q值将股票仓位调整为默认值
        order_target_value(g.security1, g.p * total_value)
        order_target_value(g.security2, g.q * total_value)
        # 代码里重复两遍因为要先卖后买，而我们没有特地确定哪个先哪个后
        order_target_value(g.security1, g.p * total_value)
        order_target_value(g.security2, g.q * total_value)
        # 状态改为‘平’
        g.state = 'even'

# ---代码块7，计算P值
# 输入是待计算的p值的股票list

def find_cointegrated_pairs(context, stock_list):
    delta = timedelta(days=g.p_value_days)
    starte_date = context.previous_date - delta
    prices_df = get_price(stock_list, start_date=starte_date, end_date=context.previous_date, frequency="daily", fields=["close"])["close"]
#history(g.p_value_days, unit='1d', field='close', security_list=None, df=True, skip_paused=False, fq='pre')
    # 得到DataFrame长度
    n = prices_df.shape[1]
    # 初始化p值矩阵
    pvalue_matrix = np.ones((n, n))
    # 抽取列的名称
    keys = prices_df.keys()
    # 初始化强协整组
    pairs = []
    # 对于每一个i
    for i in range(n):
        # 对于大于i的j
        for j in range(i+1, n):
            # 获取相应的两只股票的价格Series
            stock1 = np.log(prices_df[keys[i]])
            stock2 = np.log(prices_df[keys[j]])
            # 分析它们的协整关系
            result = sm.tsa.stattools.coint(stock1, stock2)
            # 取出并记录p值
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            # 如果p值小于0.05
            if pvalue < 0.05:
                # 记录股票对和相应的p值
                pairs.append((keys[i], keys[j], pvalue))
            else:
                continue
    # 返回结果
    return pvalue_matrix, pairs
    
# ---代码块7，计算相关系数
# 输入是待计算的股票对
def get_coef_by_ols(context, pairs):
    delta = timedelta(days=g.coef_days)
    starte_date = context.previous_date - delta
    prices_df = get_price(pairs, start_date=starte_date, end_date=context.previous_date, frequency="daily", fields=["close"])["close"]
    stock_df1 = prices_df[pairs[0]]
    stock_df2 = prices_df[pairs[1]]
    x = stock_df1
    y = stock_df2
    X = sm.add_constant(x)
    result = (sm.OLS(y,X)).fit()
    return result.params[pairs[0]]
    

# ---代码块7,选取行业最低p值股票对
# 输入证监会行业代码
def find_min_p(context,stocks):
    p,pairs = find_cointegrated_pairs(context,stocks)
    min_p = 100
    min_pair = []
    for p in pairs:
        pv = p[2]
        if  pv < min_p and pv >0.00001:
            min_p = pv
            min_pair = p
    return min_pair

# ---代码块8，过滤st,停牌股
# 输入待过滤的股票list
def paused_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not current_data[stock].paused]
    return security_list


def delisted_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not '退' in current_data[stock].name]
    return security_list

def st_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not current_data[stock].is_st]
    return security_list
    

# ---代码块9，找出表现较好的行业
# 输入需要选取的行业个数
def get_all_low_pb_by_industry(context):
    industrys = g.SW3
    s_list1 = []
    s_list2 = []
    for industry in industrys:
        stocks = get_industry_stocks((industry), date=context.previous_date)
        # print(len(stocks))
        stocks = st_filter(stocks)
        stocks = paused_filter(stocks)
        if len(stocks) < 5:#过滤股票数特别少的行业
            industrys.remove(industry)
            continue
        stocks = get_low_pb_by_industry(stocks)
        # p,pairs = find_cointegrated_pairs(context,stocks)
        min_p = find_min_p(context,stocks)
        # print('min_p', min_p)
        # print(len(pairs))
        if len(min_p) == 0:
            continue
        s_list1.append(min_p[0])
        s_list2.append(min_p[1])
    # print(len(industrys))
    # print(len(s_list2),len(s_list1))
    return s_list1,s_list2
    
# ---代码块10，行业中pb最低的两只
# 输入需要选取的行业股票list
def get_low_pb_by_industry(stocks):
    if stocks:
        #找出最低市净率的两支股票
        df = get_fundamentals(
            query(
                valuation.code,valuation.pb_ratio,valuation.pe_ratio
                ,indicator.gross_profit_margin
            ).filter(
                valuation.code.in_(stocks),  #查询股票池中的股票
            ).order_by(
                indicator.gross_profit_margin  #ascent按升序排列
            ).limit(
                g.stock_num_ind               #返回两支市净率最低的股票
            )
        )
        # security1 = df['code'][0]    #将市净率最低的股票号赋给security1
        # security2 = df['code'][1]    #将市净率次低的股票号赋给security2
        stocks = df['code'].tolist()
    return stocks


# ---代码块11，大盘指数止损
# 输入止损点和选择方法
def dp_stoploss(kernel=2, n=10, zs=0.08):
    if not g.open_stop_loss:
        return False
    '''
    方法1：当大盘N日均线(默认60日)与昨日收盘价构成“死叉”，则发出True信号
    方法2：当大盘N日内跌幅超过zs，则发出True信号
    '''
    # 止损方法1：根据大盘指数N日均线进行止损
    if kernel == 1:
        t = n+2
        hist = attribute_history('000001.XSHG', t, '1d', 'close', df=False)
        temp1 = sum(hist['close'][1:-1])/float(n)
        temp2 = sum(hist['close'][0:-2])/float(n)
        close1 = hist['close'][-1]
        close2 = hist['close'][-2]
        if (close2 > temp2) and (close1 < temp1):
            return True
        else:
            return False
    # 止损方法2：根据大盘指数跌幅进行止损
    elif kernel == 2:
        hist1 = attribute_history('000947.XSHG', n, '1d', 'close',df=False)
        #log.info("hist1['close'][-1] = " + str(hist1['close'][-1]) + " hist1['close'][0] = " + str(hist1['close'][0]) + " / = " + str(float(hist1['close'][-1]/hist1['close'][0])))
        if ((1-float(hist1['close'][-1]/hist1['close'][0])) >= zs):
            return True
        else:
            return False
            
            
#  ---代码块12，清仓
#  输入context账号信息
def clear_all(context):
    for s in context.portfolio.positions:
        order_target(s,0)
        # log.info("!!!stop loss " + s)

#  ---代码块13，获取候选股票对
#  输入context账号信息
def get_all_pairs(context):
    stock_list1 = []
    stock_list2 = []
    for i in range(len(g.stock_list1)):
        stocks = [g.stock_list1[i]] + [g.stock_list2[i]]
        find_min_p(context,stocks)
        min_p = find_min_p(context,stocks)
        # print('min_p', min_p)
        # print(len(pairs))
        if len(min_p) == 0:
            continue
        stock_list1.append(min_p[0])
        stock_list2.append(min_p[1])
    return stock_list1,stock_list2
#==============================================
# 每天开盘前调用一次
def before_trading_start(context):
    body = read_file("data/buyList.txt")
    g.stock_list3 = []#pd.read_csv(StringIO(body))['code'].tolist()
    g.days += 1
    if g.days % g.trade_freq != 1:
        return
    # print(g.days)
    # 股票池1
    g.stock_list1 = ['300618.XSHE']#['601818.XSHG','601398.XSHG','600036.XSHG']#['601318.XSHG','600519.XSHG','000333.XSHE','600036.XSHG']#['601318.XSHG','601633.XSHG','600104.XSHG','000538.XSHE','601688.XSHG','002460.XSHE','601288.XSHG']#'601288.XSHG',
    # 股票池2
    g.stock_list2 = ['603799.XSHG']#['000001.XSHE','600104.XSHG','601318.XSHG']#['600887.XSHG','000858.XSHE','000651.XSHE','002027.XSHE']#['601336.XSHG','000625.XSHE','601238.XSHG','002424.XSHE','600999.XSHG','002466.XSHE','601988.XSHG']#'601988.XSHG',
    g.stock_list1,g.stock_list2 = get_all_pairs(context)#get_all_low_pb_by_industry(context)
    # g.security1 = min_pair[0]
    # g.security2 = min_pair[1]
    # print p,pairs



# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context, data):
    if (g.stock_list3) != 0:
        for s in g.stock_list3:
            if s not in context.portfolio.positions.keys():
                order_target_value(s, context.portfolio.portfolio_value*0.1)
    
    if g.stop_days != 0:
        g.stop_days -= 1
        return
    stop = dp_stoploss()
    if stop:
        g.stop_days = 11
        clear_all(context)
        return
    if g.days % g.trade_freq != 1:
        return
    for i in range(len(g.stock_list1)):
        g.security1 = g.stock_list1[i]
        g.security2 = g.stock_list2[i]
        new_state = get_signal(context)
        change_positions(new_state,context)


def after_trading_end(context):
    if g.days % g.trade_freq != 1:
        return
