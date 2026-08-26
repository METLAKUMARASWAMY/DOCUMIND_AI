from app.legacy import dataset_loader as loader_module
from app.legacy.dataset_loader import DatasetLoader


def test_dataset_loads_cases_documents_and_evidence():
    loader = DatasetLoader()

    cases = loader.load_all_cases()
    assert cases

    case_id = cases[0]["case_id"]
    assert loader.load_case(case_id)["case_id"] == case_id
    assert len(loader.load_case_documents(case_id)) >= 4
    assert loader.load_case_entities(case_id)
    assert loader.load_case_evidence(case_id)


def test_dataset_loads_all_label_surfaces():
    loader = DatasetLoader()
    case_id = loader.load_all_cases()[0]["case_id"]

    assert loader.load_case_compliance_findings(case_id)
    assert loader.load_case_fraud_findings(case_id)
    assert loader.load_case_cross_checks(case_id)
    assert loader.load_case_workflow(case_id)
    assert loader.load_case_agents(case_id)
    assert isinstance(loader.load_case_graph(case_id), list)
    assert loader.load_case_rag_questions(case_id)
    assert loader.load_ner_records()


def test_module_level_loader_functions_match_required_contract():
    case_id = loader_module.load_all_cases()[0]["case_id"]

    assert loader_module.load_case(case_id)["case_id"] == case_id
    assert loader_module.load_case_documents(case_id)
    assert loader_module.load_case_entities(case_id)
    assert loader_module.load_case_evidence(case_id)
    assert loader_module.load_case_compliance_findings(case_id)
    assert loader_module.load_case_fraud_findings(case_id)
    assert loader_module.load_case_cross_checks(case_id)
    assert loader_module.load_case_workflow(case_id)
    assert loader_module.load_case_agents(case_id)
    assert isinstance(loader_module.load_case_graph(case_id), list)
    assert loader_module.load_case_rag_questions(case_id)
    assert loader_module.load_ner_records()
