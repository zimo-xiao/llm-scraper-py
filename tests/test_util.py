"""
Tests for util.py functions
"""

import pytest
from pydantic import BaseModel
from typing import List
from llm_scraper_py.util import dict_to_model_type


class SimpleModel(BaseModel):
    name: str
    value: int


class NestedModel(BaseModel):
    id: int
    title: str
    description: str


class ContainerModel(BaseModel):
    items: List[NestedModel]
    count: int
    metadata: dict


class Project(BaseModel):
    name: str
    description: str
    status: str


class ProjectList(BaseModel):
    projects: List[Project]
    total_count: int


class TestDictToModelType:
    def test_dict_to_model_type_simple(self):
        """Test conversion of simple dict to Pydantic model"""
        data = {"name": "test", "value": 42}
        result = dict_to_model_type(data, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_dict_to_model_type_nested_list(self):
        """Test conversion of dict with nested list of models"""
        data = {
            "items": [
                {"id": 1, "title": "First", "description": "First item"},
                {"id": 2, "title": "Second", "description": "Second item"},
            ],
            "count": 2,
            "metadata": {"source": "test"},
        }

        result = dict_to_model_type(data, ContainerModel)

        assert isinstance(result, ContainerModel)
        assert result.count == 2
        assert result.metadata == {"source": "test"}
        assert len(result.items) == 2
        assert isinstance(result.items[0], NestedModel)
        assert result.items[0].id == 1
        assert result.items[0].title == "First"
        assert result.items[1].id == 2
        assert result.items[1].title == "Second"

    def test_dict_to_model_type_project_list(self):
        """Test conversion of ProjectList with nested Project models"""
        data = {
            "projects": [
                {
                    "name": "Project Alpha",
                    "description": "First project",
                    "status": "active",
                },
                {
                    "name": "Project Beta",
                    "description": "Second project",
                    "status": "completed",
                },
                {
                    "name": "Project Gamma",
                    "description": "Third project",
                    "status": "pending",
                },
            ],
            "total_count": 3,
        }

        result = dict_to_model_type(data, ProjectList)

        assert isinstance(result, ProjectList)
        assert result.total_count == 3
        assert len(result.projects) == 3

        # Check first project
        assert isinstance(result.projects[0], Project)
        assert result.projects[0].name == "Project Alpha"
        assert result.projects[0].description == "First project"
        assert result.projects[0].status == "active"

        # Check second project
        assert isinstance(result.projects[1], Project)
        assert result.projects[1].name == "Project Beta"
        assert result.projects[1].status == "completed"

        # Check third project
        assert isinstance(result.projects[2], Project)
        assert result.projects[2].name == "Project Gamma"
        assert result.projects[2].status == "pending"

    def test_dict_to_model_type_empty_data(self):
        """Test conversion with empty data"""
        result = dict_to_model_type({}, SimpleModel)
        assert result is None

        result = dict_to_model_type(None, SimpleModel)
        assert result is None

    def test_dict_to_model_type_no_model_class(self):
        """Test conversion with no model class"""
        data = {"name": "test", "value": 42}
        result = dict_to_model_type(data, None)
        assert result is None

    def test_dict_to_model_type_non_basemodel_class(self):
        """Test conversion with non-BaseModel class"""
        data = {"name": "test", "value": 42}
        result = dict_to_model_type(data, dict)  # dict is not a BaseModel subclass
        assert result == data  # Should return original data

    def test_dict_to_model_type_validation_error(self):
        """Test conversion with invalid data that fails validation"""
        data = {"name": "test"}  # missing required 'value' field

        with pytest.raises(Exception):  # Should raise validation error
            dict_to_model_type(data, SimpleModel)

    def test_dict_to_model_type_extra_fields(self):
        """Test conversion with extra fields in data"""
        data = {"name": "test", "value": 42, "extra_field": "ignored"}
        result = dict_to_model_type(data, SimpleModel)

        assert isinstance(result, SimpleModel)
        assert result.name == "test"
        assert result.value == 42
        # extra_field should be ignored by Pydantic

    def test_dict_to_model_type_empty_nested_list(self):
        """Test conversion with empty nested list"""
        data = {"projects": [], "total_count": 0}

        result = dict_to_model_type(data, ProjectList)

        assert isinstance(result, ProjectList)
        assert result.total_count == 0
        assert len(result.projects) == 0
        assert isinstance(result.projects, list)
