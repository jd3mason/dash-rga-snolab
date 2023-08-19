from datetime import datetime
import base64
import io
from dash import Dash, dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px
import pandas as pd
import xlsxwriter

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN, dbc_css])
server = app.server
template = load_figure_template('cerulean')


upload_data = html.Div(
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or Click to Select File']),
        style={
            'width': '100%',
            'height': '11vh',
            'lineHeight': '11vh',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
        },
        className='mt-3'
    ),
)

species_dropdown = html.Div(
    dcc.Dropdown(
        id='species-dropdown',
        options=[
            {'label': 'C-12', 'value': 'Mass 12.00'},
            {'label': 'CH-13', 'value': 'Mass 13.00'},
            {'label': 'N-14', 'value': 'Mass 14.00'},
            {'label': 'N-15', 'value': 'Mass 15.00'},
            {'label': 'O-16', 'value': 'Mass 16.00'},
            {'label': 'OH-17', 'value': 'Mass 17.00'},
            {'label': 'H\u2082O-18', 'value': 'Mass 18.00'},
            {'label': 'Ar-20', 'value': 'Mass 20.00'},
            {'label': 'N\u2082-28', 'value': 'Mass 28.00'},
            {'label': 'N\u2082-29', 'value': 'Mass 29.00'},
            {'label': 'NO\u2093-30', 'value': 'Mass 30.00'},
            {'label': 'O\u2082-32', 'value': 'Mass 32.00'},
            {'label': 'Ar-36', 'value': 'Mass 36.00'},
            {'label': 'Ar-38', 'value': 'Mass 38.00'},
            {'label': 'Ar-40', 'value': 'Mass 40.00'},
            {'label': 'CO\u2082-44', 'value': 'Mass 44.00'}
        ],
        placeholder='Species',
        maxHeight=200,
        multi=True,
        className='m-3'
    )
)

PvT_graph = html.Div([
    dcc.Graph(
        id='PvT-graph',
        config={'displayModeBar': True, 'toImageButtonOptions': {'height': 675, 'width': 1101}},
        style={'height': '78.55vh'}
    )
])

spectrum_graph = html.Div([
    dcc.Graph(
        id='spectrum-graph',
        config={'displayModeBar': True, 'toImageButtonOptions': {'height': 675, 'width': 1101}},
        style={'height': '90.8vh'}
    )
])

spectrum_time = html.P(id='spectrum-time',style={'font-size': '17px'})

concentration_table = html.Div([
    dbc.Table(
        id='concentration-table',
        bordered=True
    )
])

argon_adjust_checklist = html.Div([
    dcc.Checklist(
        id='argon-adjust-checklist',
        options=[
            {'label': ' Calculate Ar-40 using Ar-36', 'value': 'calculate'},
        ],
        value=['']
    )
])

excel_button = html.Div([
    dbc.Button(
        id='excel-button',
        children='Download Excel',
        color='info',
        class_name='mt-4'
    )
])

summary_card = dbc.Card(
    [spectrum_time, concentration_table, argon_adjust_checklist, excel_button],
    body=True,
    className='mt-1 mb-1'
)

tab1 = dbc.Tab([species_dropdown, PvT_graph], label='PvT')
tab2 = dbc.Tab([spectrum_graph], label='Spectrum')
tabs = dbc.Card(dbc.Tabs([tab1, tab2]), className='mt-1 mb-1')


app.layout = dbc.Container([
    dcc.Store(id='stored-data'),
    dcc.Store(id='RGA-info'),
    dcc.Store(id='spectrum-data'),
    dcc.Store(id='spectrum-summary'),
    dcc.Download(id='excel-download'),
    dbc.Row([
        dbc.Col([summary_card, upload_data], width=3),
        dbc.Col([tabs], width=8)], className='g-0', justify='around')],
    className='dbc',
    fluid=True
)


@app.callback(
    Output(component_id='stored-data', component_property='data'),
    Output(component_id='RGA-info', component_property='data'),
    Output(component_id='species-dropdown', component_property='options'),
    Output(component_id='species-dropdown', component_property='value'),
    Input(component_id='upload-data', component_property='contents')
)
def store_data(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t', skiprows=16)
        df.drop(df.columns[[1, 2, 3]], axis=1, inplace=True)
        RGA_data = df.to_dict('records')

        header_lines = [line for i, line in enumerate(decoded.decode('utf-8').split('\r\n')) if i in range(16)]
        RGA_info = {}
        RGA_info['Measurement'] = header_lines[8].split('=')[1].strip().split(' ')[0]
        RGA_info['First mass'] = header_lines[9].split('=')[1].strip()
        RGA_info['Last mass'] = header_lines[10].split('=')[1].strip()
        RGA_info['Units'] = header_lines[12].split('=')[1].strip() 
        
        species_masses = [12, 13, 14, 15, 16, 17, 18, 20, 28, 29, 30, 32, 36, 38, 40, 44]
        species_labels = ['C-12', 'CH-13', 'N-14', 'N-15', 'O-16', 'OH-17', 'H\u2082O-18', 'Ar-20', 'N\u2082-28', 'N\u2082-29', 'NO\u2093-30', 'O\u2082-32', 'Ar-36', 'Ar-38', 'Ar-40', 'CO\u2082-44']
        species_colors = ['chocolate', 'brown', 'limegreen', 'lightgreen', 'tomato', 'blue', 'cornflowerblue', 'gold', 'mediumseagreen', 'green', 'darkturquoise', 'red', 'purple', 'pink', 'black', 'orange']
        species_values = ['Mass ' + str(mass) for mass in species_masses] if RGA_info['Measurement'] == 'Barchart' else ['Mass ' + str(mass) + '.00' for mass in species_masses]
        species_disabled = [not ((mass >= float(RGA_info['First mass'])) & (mass <= float(RGA_info['Last mass']))) for mass in species_masses]
        species_dropdown_keys = ['label', 'value', 'disabled']
        species_zipped = list(zip(species_labels, species_values, species_disabled))
        species_dropdown_options = [{species_dropdown_keys[i]: species_tuple[i] for i in range(len(species_dropdown_keys))} for species_tuple in species_zipped]

        species_values_default = []
        species_disabled_default = []
        for species in species_dropdown_options:
            if species['label'] in ['H\u2082O-18','N\u2082-28', 'O\u2082-32', 'Ar-40', 'CO\u2082-44']:
                species_values_default.append(species['value'])
                species_disabled_default.append(species['disabled'])
        species_dropdown_values = [value for value, disabled in zip(species_values_default, species_disabled_default) if not disabled]
 
        RGA_info['Species'] = [{species_dropdown_keys[i]: species_tuple[i] for i in range(len(species_dropdown_keys))} for species_tuple in species_zipped]
        for i, species in enumerate(RGA_info['Species']):
            species['color'] = species_colors[i]
            species['mass'] = species_masses[i]
    else:
        RGA_data = None
        RGA_info = {}
        species_dropdown_options = []
        species_dropdown_values = []

    return RGA_data, RGA_info, species_dropdown_options, species_dropdown_values


@app.callback(
    Output(component_id='PvT-graph', component_property='figure'),
    Input(component_id='stored-data', component_property='data'),
    Input(component_id='species-dropdown', component_property='value'),
    State(component_id='RGA-info', component_property='data')
)
def update_PvT_plot(stored_data, species_dropdown, RGA_info):
    if stored_data is not None:
        df = pd.DataFrame(stored_data) 
        df['Scan Time'] = pd.to_datetime(df['Scan Time'], format='%d/%m/%y %H:%M:%S.%f')
        df['Scan Time'] = [t.replace(microsecond=0) for t in df['Scan Time']]
        df.set_index('Scan Time', inplace=True)
        species_mass_list =[int(species.split(' ')[1]) if RGA_info['Measurement'] == 'Barchart' else float(species.split(' ')[1]) for species in species_dropdown]
        species_mass_list_sorted = sorted(species_mass_list)
        species_dropdown_sorted = ['Mass ' + str(mass) for mass in species_mass_list_sorted] if RGA_info['Measurement'] == 'Barchart' else ['Mass ' + str("{:.2f}".format(mass)) for mass in species_mass_list_sorted]
        dff = df.loc[:, species_dropdown_sorted]
        tracelabels = {species['value']: species['label'] for species in RGA_info['Species'] if species['value'] in species_dropdown_sorted}
        species_color_map = {species['value']: species['color'] for species in RGA_info['Species'] if species['value'] in species_dropdown_sorted}
        date_buttons = [{'count': 1, 'step': 'hour', 'stepmode': 'backward', 'label': '1H'},
                        {'count': 2, 'step': 'hour', 'stepmode': 'backward', 'label': '2H'},
                        {'count': 4, 'step': 'hour', 'stepmode': 'backward', 'label': '4H'},
                        {'count': 6, 'step': 'hour', 'stepmode': 'backward', 'label': '6H'},
                        {'count': 12, 'step': 'hour', 'stepmode': 'backward', 'label': '12H'},
                        {'count': 1, 'step': 'day', 'stepmode': 'backward', 'label': '1D'},
                        {'count': 3, 'step': 'day', 'stepmode': 'backward', 'label': '3D'},
                        {'count': 7, 'step': 'day', 'stepmode': 'backward', 'label': '1W'},
                        {'count': 14, 'step': 'day', 'stepmode': 'backward', 'label': '2W'},
                        {'count': 1, 'step': 'month', 'stepmode': 'backward', 'label': '1M'}]
        fig_PvT = px.line(dff, x=dff.index, y=dff.columns, log_y=True, color_discrete_map=species_color_map)
        fig_PvT.update_layout(xaxis_title='Time', yaxis_title='Pressure (' + RGA_info['Units'] + ')', legend=dict(title=None))
        fig_PvT.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
        fig_PvT.for_each_trace(lambda t: t.update(name = tracelabels[t.name], legendgroup = tracelabels[t.name], hovertemplate = t.hovertemplate.replace(t.name, tracelabels[t.name])))
    else:
        fig_PvT = px.line(x=[datetime.now()], y=[1e-6], log_y=True, range_y=[1e-5, 1000]) 
        fig_PvT.update_layout(xaxis_title='Time', yaxis_title='Pressure (mbar)')

    return fig_PvT


@app.callback(
    Output(component_id='spectrum-graph', component_property='figure'),
    Output(component_id='spectrum-time', component_property='children'),
    Output(component_id='spectrum-data', component_property='data'),
    Input(component_id='PvT-graph', component_property='clickData'),
    Input(component_id='stored-data', component_property='data'),
    State(component_id='RGA-info', component_property='data')
)
def update_spectrum_plot(clk_data, stored_data, RGA_info):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
        df['Scan Time'] = pd.to_datetime(df['Scan Time'], format='%d/%m/%y %H:%M:%S.%f')
        df['Scan Time'] = [t.replace(microsecond=0) for t in df['Scan Time']]
        df.set_index('Scan Time', inplace=True)   
        mass_list = [float(mass_string.split(sep=' ')[1]) for mass_string in df.columns]
        if clk_data is None: spectrum_time = datetime.strftime(df.index[-1], '%Y-%m-%d %H:%M:%S')
        else: spectrum_time = datetime.strftime(df.index[-1], '%Y-%m-%d %H:%M:%S') if clk_data['points'][0]['x'] not in df.index else clk_data['points'][0]['x']
        spectrum_time = datetime.strptime(spectrum_time + ':00', '%Y-%m-%d %H:%M:%S') if spectrum_time.count(':') < 2 else datetime.strptime(spectrum_time, '%Y-%m-%d %H:%M:%S')
        spectrum = df.loc[spectrum_time,:]
        fig_spectrum = px.bar(x=mass_list, y=spectrum, log_y=True, range_y=[1e-7, 1000]) if RGA_info['Measurement'] == 'Barchart' else px.line(x=mass_list, y=spectrum, log_y=True, range_y=[1e-7, 1000])  
        fig_spectrum.update_layout(xaxis_title='Mass/Charge', yaxis_title='Pressure (' + RGA_info['Units'] + ')') 
        spectrum_time_string = spectrum_time.strftime('%b %d, %Y %H:%M:%S')
        spectrum_data = spectrum.to_dict()
    else:
        fig_spectrum = px.line(x=[0], y=[1e-8], log_y=True, range_x=[1, 50],range_y=[1e-7, 1000])  
        fig_spectrum.update_layout(xaxis_title='Mass/Charge', yaxis_title='Pressure (mbar)')
        spectrum_time_string = datetime.now().strftime('%b %d, %Y %H:%M:%S')
        spectrum_data=None

    return fig_spectrum, spectrum_time_string, spectrum_data


@app.callback(
    Output(component_id='concentration-table', component_property='children'),
    Output(component_id='spectrum-summary', component_property='data'),
    Input(component_id='spectrum-data', component_property='data'),
    Input(component_id='argon-adjust-checklist', component_property='value'),
    State(component_id='RGA-info', component_property='data')
)
def update_concentration_table(spectrum_data, argon_adjust, RGA_info):
    if spectrum_data is not None:
        spectrum = pd.Series(spectrum_data)
        total_pressure = 0
        for species in RGA_info['Species']:
            if not species['disabled']:
                if species['mass'] in [20, 36, 38, 40]: total_pressure = total_pressure + spectrum[species['value']]/1.2
                elif species['mass'] == 44: total_pressure = total_pressure + spectrum[species['value']]/1.4
                else: total_pressure = total_pressure + spectrum[species['value']]

        species_columns = [species['value'] for species in RGA_info['Species'] if not species['disabled']]
        barchart = (RGA_info['Measurement'] == 'Barchart')
        if any('36' in value for value in species_columns):
            argon40 = ((spectrum['Mass 36']/1.2)*(99.6/0.334)) if barchart else ((spectrum['Mass 36.00']/1.2)*(99.6/0.334))
            if any('40' in value for value in species_columns):
                    calculated_total_pressure = total_pressure - (spectrum['Mass 40']/1.2) + argon40 if barchart else total_pressure - (spectrum['Mass 40.00']/1.2) + argon40
            else: calculated_total_pressure = total_pressure + argon40
        else:
            calculated_total_pressure = total_pressure
            if any('40' in value for value in species_columns):
                argon40 = (spectrum['Mass 40']/1.2) if barchart else (spectrum['Mass 40.00']/1.2)
            else: argon40 = 0

        species_masses, species_pressures = [], []
        for species in RGA_info['Species']:
            species_masses.append(species['mass'])
            species_pressures.append(spectrum[species['value']] if not species['disabled'] else 0)
        spectrum_summary = {'masses': species_masses, 'pressures': species_pressures}

        if 'calculate' in argon_adjust:
            carbondioxide = ((species_pressures[species_masses.index(44)]/1.4)/calculated_total_pressure)*1e6
            water = (species_pressures[species_masses.index(18)]/calculated_total_pressure)*1e6
            nitrogen = (species_pressures[species_masses.index(28)]/calculated_total_pressure)*1e6
            oxygen = (species_pressures[species_masses.index(32)]/calculated_total_pressure)*1e6
            argon = ((species_pressures[species_masses.index(40)]/1.2)/calculated_total_pressure)*1e6
            argon_purity = (((species_pressures[species_masses.index(20)]/1.2) + (species_pressures[species_masses.index(36)]/1.2) +(species_pressures[species_masses.index(38)]/1.2) + (species_pressures[species_masses.index(40)]/1.2))/calculated_total_pressure)*1e6
        else:
            carbondioxide = ((species_pressures[species_masses.index(44)]/1.4)/total_pressure)*1e6
            water = (species_pressures[species_masses.index(18)]/total_pressure)*1e6
            nitrogen = (species_pressures[species_masses.index(28)]/total_pressure)*1e6
            oxygen = (species_pressures[species_masses.index(32)]/total_pressure)*1e6
            argon = ((species_pressures[species_masses.index(40)]/1.2)/total_pressure)*1e6
            argon_purity = (((species_pressures[species_masses.index(20)]/1.2) + (species_pressures[species_masses.index(36)]/1.2) +(species_pressures[species_masses.index(38)]/1.2) + (species_pressures[species_masses.index(40)]/1.2))/total_pressure)*1e6    

        carbondioxide_string = str("{:.1f}".format(carbondioxide)) + ' ppm' if carbondioxide < 10000 else str("{:.5f}".format(carbondioxide/10000)) + ' %'
        water_string = str("{:.1f}".format(water)) + ' ppm' if water < 10000 else str("{:.5f}".format(water/10000)) + ' %'
        nitrogen_string = str("{:.1f}".format(nitrogen)) + ' ppm' if nitrogen < 10000 else str("{:.5f}".format(nitrogen/10000)) + ' %'
        oxygen_string = str("{:.1f}".format(oxygen)) + ' ppm' if oxygen < 10000 else str("{:.5f}".format(oxygen/10000)) + ' %'
        argon_string = str("{:.1f}".format(argon)) + ' ppm' if argon < 10000 else str("{:.5f}".format(argon/10000)) + ' %'
        argon_purity_string = str("{:.1f}".format(argon_purity)) + ' ppm' if argon_purity < 10000 else str("{:.5f}".format(argon_purity/10000)) + ' %'
        table = dbc.Table.from_dataframe(pd.DataFrame({'Species': ['H\u2082O-18', 'N\u2082-28', 'O\u2082-32', 'Ar-40', 'CO\u2082-44', 'Argon purity'], 'Concentration': [water_string, nitrogen_string, oxygen_string, argon_string, carbondioxide_string, argon_purity_string]}))
    else:
        table = dbc.Table.from_dataframe(pd.DataFrame({'Species': ['H\u2082O-18', 'N\u2082-28', 'O\u2082-32', 'Ar-40', 'CO\u2082-44', 'Argon purity'], 'Concentration': ['', '', '', '', '', '']}))
        spectrum_summary = {}

    return table, spectrum_summary


@app.callback(
    Output(component_id='excel-download', component_property='data'),
    Input(component_id='excel-button', component_property='n_clicks'),
    State(component_id='spectrum-time', component_property='children'),
    State(component_id='spectrum-summary', component_property='data'),
    prevent_initial_call = True
)
def generate_excel_file(n_clicks, spectrum_time, spectrum_summary):
    output = io.BytesIO()      
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    workbook = writer.book
    worksheet = workbook.add_worksheet()

    formatfont = workbook.add_format({'font_size': 12})
    formatB4 = workbook.add_format({'top':2, 'bottom': 2, 'left': 2, 'right': 1, 'font_size': 12})
    formatC4 = workbook.add_format({'top':2, 'bottom': 2, 'left': 1, 'right': 2, 'font_size': 12})
    formatB = workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatC = workbook.add_format({'left': 1, 'right': 2, 'num_format': '0.00E+00', 'font_size': 12})
    formatD = workbook.add_format({'num_format': '0.00E+00', 'font_size': 12})
    formatE = workbook.add_format({'num_format': '0.00E+00', 'font_size': 12})
    formatF = workbook.add_format({'num_format': '0.00', 'font_size': 12})
    formatB20 = workbook.add_format({'bottom': 2, 'left': 2, 'right': 1, 'font_size': 12})
    formatC20 = workbook.add_format({'bottom': 2, 'left': 1, 'right': 2, 'font_size': 12, 'num_format': '0.00E+00'})
    formatC22 = workbook.add_format({'num_format': '0.00E+00', 'font_size': 12})
    formatG10 = workbook.add_format({'num_format': '0.00', 'font_size': 12})
    worksheet.set_column('A:B', 10.42, formatfont)
    worksheet.set_column('C:C', 20)
    worksheet.set_column('D:G', 10.42)
    worksheet.write('B4', 'RGA Peak', formatB4)
    worksheet.write('C4', 'Raw Value', formatC4)
    worksheet.write('D4', 'I_Corrected', formatfont)
    worksheet.write('E4', 'Fraction', formatfont)
    worksheet.write('F4', 'PPM', formatfont)
    worksheet.write('G4', 'Degen', formatfont)
    for row in range(15):
        formula_string_D = 'C' + str(row+5)
        formula_string_E = 'D' + str(row+5) + '/D$22'
        formula_string_F = 'E' + str(row+5) + '*1000000'
        worksheet.write(row+4, 1, spectrum_summary['masses'][row], formatB)
        worksheet.write(row+4, 2, spectrum_summary['pressures'][row], formatC)
        worksheet.write_formula(row+4, 3, formula_string_D, formatD)
        worksheet.write_formula(row+4, 4, formula_string_E, formatE)
        worksheet.write_formula(row+4, 5, formula_string_F, formatF)
    worksheet.write('B20', spectrum_summary['masses'][-1], formatB20)
    worksheet.write('C20', spectrum_summary['pressures'][-1], formatC20)
    worksheet.write_formula('E20', '=D20/D$22', formatE)
    worksheet.write_formula('F20', '=E20*1000000', formatF)
    worksheet.write_formula('D12', '=C12/1.2', formatD)
    worksheet.write_formula('D17', '=C17/1.2', formatD)
    worksheet.write_formula('D18', '=C18/1.2', formatD)
    worksheet.write_formula('D19', '=C19/1.2', formatD)
    worksheet.write_formula('D20', '=C20/1.4', formatD)
    worksheet.write_formula('G9', '=F9-(0.02*F11)', formatfont)
    worksheet.write_formula('G10', '=F10-(0.25*F11)', formatG10)
    worksheet.write('B22', 'Sum', formatfont)
    worksheet.write_formula('C22', '=SUM(C5:C20)', formatC22)
    worksheet.write_formula('D22', '=SUM(D5:D20)', formatD)
    worksheet.write_formula('E22', '=SUM(E5:E20)', formatE)
    worksheet.write_formula('F22', '=SUM(F5:F20)', formatF)
        
    formatJ5= workbook.add_format({'top':2, 'bottom': 2, 'left': 2, 'right': 1, 'font_size': 12})
    formatJ6= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ7= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ8= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ9= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ10= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ11= workbook.add_format({'left': 2, 'right': 1, 'font_size': 12})
    formatJ12= workbook.add_format({'top':2, 'bottom': 2, 'left': 2, 'right': 2, 'font_size': 12})
    formatK5= workbook.add_format({'top':2, 'bottom': 2, 'left': 1, 'right': 2, 'font_size': 12})
    formatK6= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12})
    formatK7= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12})
    formatK8= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12, 'num_format': '0.00'})
    formatK9= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12, 'num_format': '0.00'})
    formatK10= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12, 'num_format': '0.00'})
    formatK11= workbook.add_format({'left': 1, 'right': 2, 'font_size': 12, 'num_format': '0.00'})
    formatK12= workbook.add_format({'top':2, 'bottom': 2, 'left': 2, 'right': 2, 'font_size': 12, 'num_format': '0.00'})
    worksheet.set_column('H:L', 10.42)
    worksheet.write('J5', 'Chemical', formatJ5)
    worksheet.write('J6', 'CH4', formatJ6)
    worksheet.write('J7', 'CO', formatJ7)
    worksheet.write('J8', 'CO2', formatJ8)
    worksheet.write('J9', 'H2O', formatJ9)
    worksheet.write('J10', 'N2', formatJ10)
    worksheet.write('J11', 'O2', formatJ11)
    worksheet.write('J12', 'Ar-40', formatJ12)
    worksheet.write('K5', 'PPM level', formatK5)
    worksheet.write_formula('K6', '=(G9+F5)/2', formatK6)
    worksheet.write_formula('K7', '=(G9+F5)/2', formatK7)
    worksheet.write_formula('K8', '=F20', formatK8)
    worksheet.write_formula('K9', '=F11', formatK9)
    worksheet.write_formula('K10', '=F13', formatK10)
    worksheet.write_formula('K11', '=F16', formatK11)
    worksheet.write_formula('K12', '=F19', formatK12)
        
    formatJ14= workbook.add_format({'top':2, 'bottom': 2, 'left': 2, 'font_size': 12})
    formatJ15= workbook.add_format({'left': 2, 'font_size': 12})
    formatJ16= workbook.add_format({'left': 2, 'font_size': 12})
    formatJ17= workbook.add_format({'left': 2, 'font_size': 12})
    formatJ18= workbook.add_format({'left': 2, 'font_size': 12})
    formatJ19= workbook.add_format({'bottom': 2, 'left': 2, 'font_size': 12})
    formatK14= workbook.add_format({'top': 2, 'bottom': 2, 'font_size': 12})
    formatK15= workbook.add_format({'num_format': '0.0000', 'font_size': 12})
    formatK16= workbook.add_format({'num_format': '0.0000', 'font_size': 12})
    formatK17= workbook.add_format({'num_format': '0.0000', 'font_size': 12})
    formatK19= workbook.add_format({'bottom': 2, 'num_format': '0.00000', 'font_size': 12})
    formatL14= workbook.add_format({'top':2, 'bottom': 2, 'right': 2, 'font_size': 12})
    formatL15= workbook.add_format({'right': 2, 'font_size': 12})
    formatL16= workbook.add_format({'right': 2, 'font_size': 12})
    formatL17= workbook.add_format({'right': 2, 'font_size': 12})
    formatL18= workbook.add_format({'right': 2, 'font_size': 12})
    formatL19= workbook.add_format({'bottom': 2, 'right': 2, 'font_size': 12})
    worksheet.write('J14', 'Argon', formatJ14)
    worksheet.write('J15', 36, formatJ15)
    worksheet.write('J16', 38, formatJ16)
    worksheet.write('J17', 40, formatJ17)
    worksheet.write('J18', '', formatJ18)
    worksheet.write('J19', 'Total Purity', formatJ19)
    worksheet.write('K14', '', formatK14)
    worksheet.write_formula('K15', '=100*(C17/(C$19+C$18+C$17+C$12))', formatK15)
    worksheet.write_formula('K16', '=100*(C18/(C$19+C$18+C$17+C$12))', formatK16)
    worksheet.write_formula('K17', '=100*(C19/(C$19+C$18+C$17+C$12))', formatK17)
    worksheet.write_formula('K19', '=100*(E17+E18+E19+E12)', formatK19)
    worksheet.write('L14', '', formatL14)
    worksheet.write('L15', '%', formatL15)
    worksheet.write('L16', '%', formatL16)
    worksheet.write('L17', '%', formatL17)
    worksheet.write('L18', '', formatL18)
    worksheet.write('L19', '%', formatL19)

    writer.close()
    data = output.getvalue()

    spectrum_datetime = datetime.strptime(spectrum_time, '%b %d, %Y %H:%M:%S')
    spectrum_time_string = spectrum_datetime.strftime('%Y-%m-%d_%H%M')
    excel_filename = 'RGAScanPurity_' + spectrum_time_string + '.xlsx'
    
    return dcc.send_bytes(data, excel_filename)

   

if __name__=='__main__':
    app.run_server(debug=False)
