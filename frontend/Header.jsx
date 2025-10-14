import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import optiuLogo from "/optiu2.png";
import { FiChevronDown, FiChevronUp } from 'react-icons/fi';
import { useAuth } from './contexts/AuthContext';

const headerSize = "8vh";

const Header = () => {
    const location = useLocation();
    const { user, logout } = useAuth();
    const isAdmin = !!user && ((user.company_name || user.company || '').toLowerCase() === 'admin');
    
    const isActive = (path) => location.pathname === path ? 'text-blue-800 border-b-2 border-blue-800 pb-1' : 'text-blue-600';
    
    // Check if any of the dropdown items are active
    const isDropdownItemActive = ['/module1', '/module1/orders', '/module1/history'].includes(location.pathname);
    const isDropdownItemActive3 = ['/module6', '/'].includes(location.pathname);
    const isDropdownItemActive4 = ['/module4', '/module4/planning', '/', '/planning'].includes(location.pathname);
    const isDropdownItemActive5 = ['/module3', '/module3/customer'].includes(location.pathname);

    return (
        //bottom heavy shadow
        <header className=" top-0 left-0 h-[8vh] flex items-center justify-center px-6 py-4 bg-blue-50 shadow-md w-full border-b-2 border-blue-200 rounded-b-3xl">
            <div className="w-full max-w-7xl mx-auto flex items-center">
                {/* <span className="text-3xl font-bold text-blue-700">OptiU</span> */}
                {/* Extra large logo */}
                <div className="w-20 flex-shrink-0 flex items-center mr-2 -my-2">
                    <img src={optiuLogo} alt="OptiU" className="h-full w-auto" />
                </div>
                
                {/* Right-aligned navigation */}
                <div className="ml-auto flex items-center space-x-6">
                    {/* Admin: show full header with all modules */}
                    {isAdmin ? (
                        <>
                            {/* Systems dropdown (Module 1 entries) */}
                            <div className="relative group">
                                <button 
                                    className={`font-bold text-sm flex items-center ${isDropdownItemActive ? 'text-blue-800 border-b-2 border-blue-800 pb-1' : 'text-blue-600'}`}
                                >
                                    Systems
                                    <FiChevronDown className="ml-1 transition-transform duration-300 ease-in-out group-hover:rotate-180" />
                                </button>
                                <div className="absolute left-1/2 -translate-x-1/2 pt-2 w-max bg-transparent transition-opacity duration-300 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto z-50">
                                   <div className="bg-white rounded-md shadow-lg py-1">
                                        <Link to="/module1" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Order Scheduler
                                        </Link>
                                        <Link to="/module1/orders" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Order History
                                        </Link>
                                        <Link to="/module1/history" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Sim History
                                        </Link>
                                    </div>
                                </div>
                            </div>
                            <Link to="/module2" className={`font-bold text-sm ${isActive('/module2')}`}>
                                PO Inject
                            </Link>
                            {/* Replenishment Orchestrator (Module 3) */}
                            <div className="relative group">
                                <button 
                                    className={`font-bold text-sm flex items-center ${isDropdownItemActive5 ? 'text-blue-800 border-b-2 border-blue-800 pb-1' : 'text-blue-600'}`}
                                >
                                    Replenish
                                    <FiChevronDown className="ml-1 transition-transform duration-300 ease-in-out group-hover:rotate-180" />
                                </button>
                                <div className="absolute left-1/2 -translate-x-1/2 pt-2 w-max bg-transparent transition-opacity duration-300 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto z-50">
                                   <div className="bg-white rounded-md shadow-lg py-1">
                                        <Link to="/module3" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Supply Location
                                        </Link>
                                        <Link to="/module3/customer" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Customer
                                        </Link>
                                    </div>
                                </div>
                            </div>
                            {/* Module 4 */}
                            <div className="relative group">
                                <button 
                                    className={`font-bold text-sm flex items-center ${isDropdownItemActive4 ? 'text-blue-800 border-b-2 border-blue-800 pb-1' : 'text-blue-600'}`}
                                >
                                    Order Fulfillment Checker
                                    <FiChevronDown className="ml-1 transition-transform duration-300 ease-in-out group-hover:rotate-180" />
                                </button>
                                <div className="absolute left-1/2 -translate-x-1/2 pt-2 w-max bg-transparent transition-opacity duration-300 opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto z-50">
                                   <div className="bg-white rounded-md shadow-lg py-1">
                                        <Link to="/module4" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Allocation Maximizer
                                        </Link>
                                        <Link to="/module4/planning" className="block whitespace-nowrap px-5 py-2 text-sm text-gray-700 hover:bg-blue-50">
                                            Planning
                                        </Link>
                                    </div>
                                </div>
                            </div>
                            <Link to="/module5" className={`font-bold text-sm ${isActive('/module5')}`}>
                                Resilience
                            </Link>
                            {/* Allocation Maximizer always visible too */}
                            <Link to="/" className={`font-bold text-sm text-center ${isActive('/')}`}>
                                Allocation Maximizer
                            </Link>
                        </>
                    ) : (
                        // Non-admin: only Allocation Maximizer
                        <>
                            <Link to="/" className={`font-bold text-sm text-center ${isActive('/')}`}>
                                Allocation Maximizer
                            </Link>
                        </>
                    )}
                    {/* Right side: auth controls */}
                    {user ? (
                        <div className="flex items-center border-l border-blue-200 pl-3 text-center">
                            <button
                                onClick={logout}
                                className="text-blue-600 text-lg hover:text-blue-800"
                            >
                                <i className="fa-solid fa-right-from-bracket"></i>
                            </button>
                        </div>
                    ) : (
                        <div className="flex items-center space-x-3 border-l border-blue-200 pl-3 text-center">
                            <Link to="/login" className="text-blue-600 text-sm font-bold hover:text-blue-800">
                                Login
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}

export default Header