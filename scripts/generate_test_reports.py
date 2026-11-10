"""
Test Blood Report PDF Generator for MediExplain AI.
Generates authentic clinical laboratory test reports for system demonstration.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

os.makedirs('data/sample_reports', exist_ok=True)

def generate_report(filename, patient_info, tests):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    # Title Style
    title_style = ParagraphStyle(
        'HospitalTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1,
        spaceAfter=3
    )
    sub_style = ParagraphStyle(
        'HospitalSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#64748B'),
        alignment=1,
        spaceAfter=10
    )

    story.append(Paragraph('METROPOLITAN CLINICAL DIAGNOSTICS & HOSPITAL LABORATORY', title_style))
    story.append(Paragraph('ISO 15189:2022 Medical Testing Accredited Laboratory | Colombo, Sri Lanka', sub_style))
    story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1E3A8A'), spaceAfter=12))

    # Patient Demographic Box
    demo_data = [
        [
            Paragraph(f'<b>Patient Name:</b> {patient_info["name"]}', styles['Normal']),
            Paragraph(f'<b>Lab Ref No:</b> {patient_info["ref_no"]}', styles['Normal']),
            Paragraph(f'<b>Date:</b> {patient_info["date"]}', styles['Normal'])
        ],
        [
            Paragraph(f'<b>Age / Gender:</b> {patient_info["age"]}, {patient_info["gender"]}', styles['Normal']),
            Paragraph(f'<b>Referred By:</b> {patient_info["doctor"]}', styles['Normal']),
            Paragraph(f'<b>Specimen:</b> Fasting Blood (EDTA / Serum)', styles['Normal'])
        ],
        [
            Paragraph(f'<b>Clinical History:</b> {patient_info["history"]}', styles['Normal']),
            '',
            ''
        ]
    ]
    demo_table = Table(demo_data, colWidths=[200, 180, 160])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('SPAN', (0,2), (2,2)),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 14))

    # Test Results Section
    sec_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )
    story.append(Paragraph('Comprehensive Metabolic & Organ Function Panel', sec_style))

    table_data = [['Test Parameter / Biomarker', 'Patient Result', 'Units', 'Reference Interval', 'Flag / Status']]
    for t in tests:
        flag_color = '#991B1B' if t['flag'] in ['HIGH', 'CRITICAL'] else ('#854D0E' if t['flag'] == 'LOW' else '#166534')
        flag_p = Paragraph(f'<font color="{flag_color}"><b>{t["flag"]}</b></font>', styles['Normal'])
        table_data.append([
            Paragraph(f'<b>{t["name"]}</b>', styles['Normal']),
            Paragraph(f'<b>{t["val"]}</b>', styles['Normal']),
            t['unit'],
            t['ref'],
            flag_p
        ])

    test_table = Table(table_data, colWidths=[180, 95, 75, 110, 80])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,1), (-1,-1), 4.5),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 16))

    # Footer Disclaimer & Signature
    story.append(HRFlowable(width='100%', thickness=0.8, color=colors.HexColor('#E2E8F0'), spaceAfter=6))
    foot_p = Paragraph(
        '<i>* Methodology: HbA1c by HPLC (NGSP certified), Glucose by Hexokinase, Creatinine by Enzymatic traceable to IDMS.<br/>'
        'Report validated by: Dr. N. Jayawardena, MD, FRCPath (Consultant Chemical Pathologist).</i>',
        ParagraphStyle('Foot', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748B'), leading=11)
    )
    story.append(foot_p)

    doc.build(story)
    print(f'[SUCCESS] Generated: {filename}')


if __name__ == '__main__':
    # Report 1: Multi-Condition Patient (Mr. K. Perera - Uncontrolled Diabetes + Renal Risk)
    generate_report(
        'data/sample_reports/Patient_01_MultiCondition_Mr_K_Perera.pdf',
        {
            'name': 'Mr. K. Perera',
            'ref_no': 'MED-2026-89412',
            'date': '2026-10-24',
            'age': '48 Yrs',
            'gender': 'Male',
            'doctor': 'Dr. S. Alwis (Consultant Physician)',
            'history': 'Uncontrolled Type 2 Diabetes, Routine Renal & Lipid Evaluation'
        },
        [
            {'name': 'HbA1c (Glycated Hb)', 'val': '8.2', 'unit': '%', 'ref': '< 5.7 %', 'flag': 'HIGH'},
            {'name': 'Fasting Blood Sugar (FBS)', 'val': '162', 'unit': 'mg/dL', 'ref': '70 - 99 mg/dL', 'flag': 'HIGH'},
            {'name': 'LDL Cholesterol', 'val': '174', 'unit': 'mg/dL', 'ref': '< 100 mg/dL', 'flag': 'HIGH'},
            {'name': 'HDL Cholesterol', 'val': '36', 'unit': 'mg/dL', 'ref': '>= 40 mg/dL', 'flag': 'LOW'},
            {'name': 'Serum Triglycerides', 'val': '220', 'unit': 'mg/dL', 'ref': '< 150 mg/dL', 'flag': 'HIGH'},
            {'name': 'Serum Creatinine', 'val': '1.8', 'unit': 'mg/dL', 'ref': '0.7 - 1.3 mg/dL', 'flag': 'HIGH'},
            {'name': 'eGFR Filtration', 'val': '44', 'unit': 'mL/min', 'ref': '>= 90 mL/min', 'flag': 'LOW'},
            {'name': 'ALT (SGPT) Liver', 'val': '88', 'unit': 'U/L', 'ref': '7 - 56 U/L', 'flag': 'HIGH'},
        ]
    )

    # Report 2: Healthy Routine Checkup (Mrs. S. Fernando - All Normal Ranges)
    generate_report(
        'data/sample_reports/Patient_02_HealthyCheckup_Mrs_S_Fernando.pdf',
        {
            'name': 'Mrs. S. Fernando',
            'ref_no': 'MED-2026-90214',
            'date': '2026-10-28',
            'age': '39 Yrs',
            'gender': 'Female',
            'doctor': 'Dr. M. Wickramatunga (GP)',
            'history': 'Routine Annual Health Screening. Asymptomatic.'
        },
        [
            {'name': 'HbA1c (Glycated Hb)', 'val': '5.2', 'unit': '%', 'ref': '< 5.7 %', 'flag': 'NORMAL'},
            {'name': 'Fasting Blood Sugar (FBS)', 'val': '88', 'unit': 'mg/dL', 'ref': '70 - 99 mg/dL', 'flag': 'NORMAL'},
            {'name': 'LDL Cholesterol', 'val': '84', 'unit': 'mg/dL', 'ref': '< 100 mg/dL', 'flag': 'NORMAL'},
            {'name': 'HDL Cholesterol', 'val': '58', 'unit': 'mg/dL', 'ref': '>= 50 mg/dL', 'flag': 'NORMAL'},
            {'name': 'Serum Triglycerides', 'val': '112', 'unit': 'mg/dL', 'ref': '< 150 mg/dL', 'flag': 'NORMAL'},
            {'name': 'Serum Creatinine', 'val': '0.8', 'unit': 'mg/dL', 'ref': '0.6 - 1.1 mg/dL', 'flag': 'NORMAL'},
            {'name': 'eGFR Filtration', 'val': '104', 'unit': 'mL/min', 'ref': '>= 90 mL/min', 'flag': 'NORMAL'},
            {'name': 'ALT (SGPT) Liver', 'val': '22', 'unit': 'U/L', 'ref': '7 - 45 U/L', 'flag': 'NORMAL'},
        ]
    )

    # Report 3: Critical Diabetes & Acute Renal Alert (Mr. D. Wickramasinghe - High Risk Alert)
    generate_report(
        'data/sample_reports/Patient_03_CriticalRenal_Mr_D_Wickramasinghe.pdf',
        {
            'name': 'Mr. D. Wickramasinghe',
            'ref_no': 'MED-2026-91508',
            'date': '2026-11-02',
            'age': '62 Yrs',
            'gender': 'Male',
            'doctor': 'Dr. R. Peiris (Nephrology Clinic)',
            'history': 'Severe Longstanding Diabetes, Bilateral Pedal Edema, Stage 4 CKD Monitoring'
        },
        [
            {'name': 'HbA1c (Glycated Hb)', 'val': '9.8', 'unit': '%', 'ref': '< 5.7 %', 'flag': 'CRITICAL'},
            {'name': 'Fasting Blood Sugar (FBS)', 'val': '224', 'unit': 'mg/dL', 'ref': '70 - 99 mg/dL', 'flag': 'CRITICAL'},
            {'name': 'LDL Cholesterol', 'val': '182', 'unit': 'mg/dL', 'ref': '< 100 mg/dL', 'flag': 'HIGH'},
            {'name': 'HDL Cholesterol', 'val': '32', 'unit': 'mg/dL', 'ref': '>= 40 mg/dL', 'flag': 'LOW'},
            {'name': 'Serum Triglycerides', 'val': '310', 'unit': 'mg/dL', 'ref': '< 150 mg/dL', 'flag': 'HIGH'},
            {'name': 'Serum Creatinine', 'val': '2.6', 'unit': 'mg/dL', 'ref': '0.7 - 1.3 mg/dL', 'flag': 'CRITICAL'},
            {'name': 'eGFR Filtration', 'val': '24', 'unit': 'mL/min', 'ref': '>= 90 mL/min', 'flag': 'CRITICAL'},
            {'name': 'ALT (SGPT) Liver', 'val': '94', 'unit': 'U/L', 'ref': '7 - 56 U/L', 'flag': 'HIGH'},
        ]
    )
